package com.inspiredandroid.kai.inference

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.tool
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlin.time.Duration.Companion.milliseconds

val MODEL_CATALOG = listOf(
    LocalModel(
        id = "gemma-4-e2b-it",
        displayName = "Gemma 4 E2B IT",
        fileName = "gemma-4-E2B-it.litertlm",
        sizeBytes = 2_580_000_000L,
        downloadUrl = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
        gpuMemoryMb = 676,
        defaultContextTokens = 4_096,
        maxContextTokens = 32_768,
        kvPerTokenBytes = 50_000,
        isRecommended = true,
    ),
    LocalModel(
        id = "gemma-4-e4b-it",
        displayName = "Gemma 4 E4B IT",
        fileName = "gemma-4-E4B-it.litertlm",
        sizeBytes = 3_650_000_000L,
        downloadUrl = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
        gpuMemoryMb = 710,
        defaultContextTokens = 4_096,
        maxContextTokens = 32_768,
        kvPerTokenBytes = 75_000,
    ),
    LocalModel(
        id = "qwen3-0.6b",
        displayName = "Qwen3 0.6B",
        fileName = "Qwen3-0.6B.litertlm",
        sizeBytes = 614_236_160L,
        downloadUrl = "https://huggingface.co/litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm",
        gpuMemoryMb = 300,
        defaultContextTokens = 4_096,
        maxContextTokens = 32_768,
        kvPerTokenBytes = 35_000,
    ),
)

class LiteRTInferenceEngine : LocalInferenceEngine {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var downloadJob: Job? = null
    private var idleReleaseJob: Job? = null

    private var engine: Engine? = null
    private var conversation: com.google.ai.edge.litertlm.Conversation? = null
    override var currentModelId: String? = null
        private set
    private var currentContextTokens: Int = 0

    private val _engineState = MutableStateFlow(EngineState.UNINITIALIZED)
    override val engineState: StateFlow<EngineState> = _engineState

    private val _downloadingModelId = MutableStateFlow<String?>(null)
    override val downloadingModelId: StateFlow<String?> = _downloadingModelId

    private val _downloadProgress = MutableStateFlow<Float?>(null)
    override val downloadProgress: StateFlow<Float?> = _downloadProgress

    private val _downloadError = MutableStateFlow<DownloadError?>(null)
    override val downloadError: StateFlow<DownloadError?> = _downloadError

    override suspend fun initialize(model: DownloadedModel, contextTokens: Int) {
        withContext(Dispatchers.IO) {
            idleReleaseJob?.cancel()
            if (currentModelId == model.id && currentContextTokens == contextTokens && _engineState.value == EngineState.READY) return@withContext
            _engineState.value = EngineState.INITIALIZING
            try {
                val modelFile = File(model.filePath)
                if (!modelFile.exists() || modelFile.length() < 1_000_000) {
                    throw IllegalStateException("Model file missing or too small: ${model.filePath}")
                }

                // Release any currently-loaded engine before measuring available memory,
                // otherwise its GPU/CPU working set counts against the headroom check and
                // switching between models spuriously fails (e.g. Qwen -> Gemma 4).
                val hadExistingEngine = engine != null
                release()
                _engineState.value = EngineState.INITIALIZING

                if (hadExistingEngine) {
                    // engine.close() returns before the OpenCL driver actually reclaims the
                    // previous model's GPU buffers, so loading a second model on top would
                    // briefly hold both resident and trip Android's LMK. Give the driver a
                    // beat to drain before allocating ~GB of new GPU buffers.
                    System.gc()
                    delay(GPU_DRAIN_DELAY_MS.milliseconds)
                }

                val availMem = getAvailableMemoryBytes()
                if (availMem < MIN_MEMORY_HEADROOM_BYTES) {
                    throw InsufficientMemoryException()
                }

                fun initWithBackend(backend: Backend, maxTokens: Int?): Engine {
                    val config = EngineConfig(
                        modelPath = model.filePath,
                        backend = backend,
                        cacheDir = getModelCacheDirectory(),
                        maxNumTokens = maxTokens,
                    )
                    val e = Engine(config)
                    e.initialize()
                    return e
                }

                val requestedTokens = if (contextTokens > 0) contextTokens else null
                println("LiteRT: initializing model=${model.id} maxNumTokens=$requestedTokens")

                val newEngine = try {
                    try {
                        initWithBackend(Backend.GPU(), requestedTokens)
                    } catch (e: Exception) {
                        initWithBackend(Backend.CPU(), requestedTokens)
                    }
                } catch (e: Exception) {
                    // Context size not supported — retry with model default
                    println("LiteRT: init failed with maxNumTokens=$requestedTokens, falling back to default: ${e.message}")
                    if (requestedTokens != null) {
                        try {
                            initWithBackend(Backend.GPU(), null)
                        } catch (e2: Exception) {
                            initWithBackend(Backend.CPU(), null)
                        }
                    } else {
                        throw e
                    }
                }

                engine = newEngine
                conversation = newEngine.createConversation()
                currentModelId = model.id
                currentContextTokens = contextTokens
                _engineState.value = EngineState.READY
            } catch (e: Exception) {
                _engineState.value = EngineState.ERROR
                throw e
            }
        }
    }

    override suspend fun release() {
        withContext(Dispatchers.IO) {
            // Null before close so a concurrent release() sees null and skips —
            // Conversation.close() / Engine.close() throw IllegalStateException on double-close.
            val convToClose = conversation
            val engineToClose = engine
            conversation = null
            engine = null
            currentModelId = null
            _engineState.value = EngineState.UNINITIALIZED
            runCatching { convToClose?.close() }
            runCatching { engineToClose?.close() }
        }
    }

    override fun releaseInBackground() {
        idleReleaseJob?.cancel()
        idleReleaseJob = scope.launch { release() }
    }

    override suspend fun chat(
        messages: List<InferenceMessage>,
        systemPrompt: String?,
        tools: List<LocalTool>,
    ): LocalChatResult = withContext(Dispatchers.IO) {
        idleReleaseJob?.cancel()
        try {
            val currentEngine = engine ?: throw IllegalStateException("Engine not initialized")

            val lastUserIndex = messages.indexOfLast { it.role == "user" }
            if (lastUserIndex < 0) throw IllegalStateException("No user message found")

            val sanitizedSystemPrompt = sanitizeForLiteRt(systemPrompt)
            val initialMessages = messages.subList(0, lastUserIndex).map { msg ->
                val sanitized = sanitizeForLiteRt(msg.content) ?: ""
                when (msg.role) {
                    "user" -> Message.user(sanitized)
                    else -> Message.model(sanitized)
                }
            }

            println("LiteRT: tools=${tools.map { it.name }}")
            // automaticToolCalling=false: the library injects <tools> schemas via its
            // native chat-template processor (special tokens positioned correctly), but
            // does NOT call OpenApiTool.execute() internally. Our manual loop below
            // handles execution after sendMessage() returns, so network tools like
            // web_search can take as long as they need without hitting the engine's
            // internal task-pool deadline (DEADLINE_EXCEEDED).
            val toolProviders = tools.map { tool(LocalToolOpenApiAdapter(it)) }
            val config = ConversationConfig(
                systemInstruction = sanitizedSystemPrompt?.let { Contents.of(it) },
                initialMessages = initialMessages,
                tools = toolProviders,
                samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 0.8),
                automaticToolCalling = false,
            )
            val prev = conversation
            conversation = null
            runCatching { prev?.close() }
            val conv = currentEngine.createConversation(config)
            conversation = conv

            val lastMessage = sanitizeForLiteRt(messages[lastUserIndex].content) ?: ""
            var firstReasoning: String? = null
            var nextMessage: Message = Message.user(lastMessage)

            for (iteration in 0 until MAX_TOOL_ITERATIONS) {
                val raw = try {
                    withTimeout(INFERENCE_TIMEOUT_MS.milliseconds) {
                        conv.sendMessage(nextMessage).toString()
                    }
                } catch (e: TimeoutCancellationException) {
                    throw InferenceTimeoutException()
                }
                println("LiteRT: response length=${raw.length}, hasThink=${raw.contains("<think>")} iteration=$iteration")

                if (firstReasoning == null) {
                    firstReasoning = THINK_BLOCK_REGEX.find(raw)?.groupValues?.get(1)?.trim()?.ifBlank { null }
                }
                val text = stripThinkBlocks(raw)

                println("LiteRT: text preview=${text.take(300).replace("\n", "\\n")}")
                val toolCall = if (tools.isNotEmpty()) parseFirstToolCall(text) else null
                if (toolCall == null) {
                    return@withContext LocalChatResult(content = text, reasoningContent = firstReasoning)
                }

                val localTool = tools.find { it.name == toolCall.name }
                val toolResult = if (localTool != null) {
                    println("LiteRT: tool call → ${toolCall.name}(${toolCall.arguments})")
                    val result = runBlocking { localTool.execute(toolCall.arguments) }
                    println("LiteRT: tool result ← ${result.take(200)}")
                    result
                } else {
                    """{"error":"unknown tool '${toolCall.name}'"}"""
                }

                nextMessage = Message.tool(Contents.of(Content.ToolResponse(toolCall.name, toolResult)))
            }

            LocalChatResult(
                content = "Unable to complete the request (tool loop limit reached).",
                reasoningContent = firstReasoning,
            )
        } finally {
            scheduleIdleRelease()
        }
    }

    private class LocalToolOpenApiAdapter(private val localTool: LocalTool) : OpenApiTool {
        override fun getToolDescriptionJsonString(): String = localTool.descriptionJsonString
        override fun execute(paramsJsonString: String): String {
            println("LiteRT: tool call → ${localTool.name}($paramsJsonString)")
            val result = runBlocking { localTool.execute(paramsJsonString) }
            println("LiteRT: tool result ← ${result.take(200)}")
            return result
        }
    }

    /**
     * Drops UTF-16 surrogate halves from the string. The litert-lm JNI layer passes
     * strings to the native runtime as *modified* UTF-8, which encodes supplementary-plane
     * characters (U+10000–U+10FFFF — most emoji like 🗺️, 🎉, 🔥) as surrogate-pair
     * sequences where each half becomes a 3-byte block. That is invalid as *standard*
     * UTF-8, and the native runtime's `nlohmann::json` parser crashes with "ill-formed
     * UTF-8 byte" the moment it hits one.
     *
     * Filtering surrogates drops every supplementary character (both halves are surrogate
     * code units in UTF-16) while leaving BMP characters — including BMP-only emoji like
     * ⚔️, ♻️, ❤️, and all CJK / extended Latin / accented characters — untouched.
     * No-op for strings that don't contain any supplementary character.
     */
    private fun sanitizeForLiteRt(s: String?): String? {
        if (s == null) return null
        if (s.none { it.isSurrogate() }) return s
        return s.filter { !it.isSurrogate() }
    }

    // Qwen3 emits a <think>…</think> block as part of its chat template; strip it before
    // the user sees it. Quantized Qwen3 can output nested/extra tags, so also scrub any
    // remaining bare <think>/<think> after regex replacement. Safe for Gemma 4.
    private fun stripThinkBlocks(s: String): String =
        THINK_BLOCK_REGEX.replace(s, "")
            .replace("<think>", "")
            .replace("</think>", "")
            .trim()

    private fun scheduleIdleRelease() {
        idleReleaseJob?.cancel()
        idleReleaseJob = scope.launch {
            delay(IDLE_RELEASE_MS.milliseconds)
            release()
        }
    }

    private data class ParsedToolCall(val name: String, val arguments: String)

    private fun parseFirstToolCall(text: String): ParsedToolCall? {
        val start = text.indexOf("<tool_call>")
        if (start < 0) return null
        val end = text.indexOf("</tool_call>", start + 11)
        val inner = (if (end >= 0) text.substring(start + 11, end) else text.substring(start + 11)).trim()
        if (inner.isEmpty() || !inner.startsWith("{")) return null
        return try {
            val obj = lenientJson.parseToJsonElement(inner).jsonObject
            val name = obj["name"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() } ?: return null
            val argsElem = obj["arguments"] ?: obj["parameters"]
            val args = if (argsElem is JsonObject) argsElem.toString() else "{}"
            ParsedToolCall(name, args)
        } catch (_: Throwable) {
            null
        }
    }

    companion object {
        private const val IDLE_RELEASE_MS = 5L * 60 * 1000 // 5 minutes
        private const val INFERENCE_TIMEOUT_MS = 120_000L // 2 minutes
        private const val MAX_TOOL_ITERATIONS = 6
        private const val MIN_MEMORY_HEADROOM_BYTES = 512L * 1024 * 1024 // 512 MB
        private const val DOWNLOAD_SPACE_BUFFER_BYTES = 500L * 1024 * 1024 // 500 MB
        private const val GPU_DRAIN_DELAY_MS = 750L
        private val THINK_BLOCK_REGEX = Regex("<think>(.*?)</think>", RegexOption.DOT_MATCHES_ALL)
        private val lenientJson = Json { ignoreUnknownKeys = true; isLenient = true }
    }

    override fun getDownloadedModels(): List<DownloadedModel> {
        val modelsDir = File(getModelStorageDirectory())
        if (!modelsDir.exists()) return emptyList()
        return MODEL_CATALOG.mapNotNull { catalogModel ->
            val modelDir = File(modelsDir, catalogModel.id)
            val modelFile = File(modelDir, catalogModel.fileName)
            if (modelFile.exists()) {
                DownloadedModel(
                    id = catalogModel.id,
                    displayName = catalogModel.displayName,
                    filePath = modelFile.absolutePath,
                    sizeBytes = modelFile.length(),
                )
            } else {
                null
            }
        }
    }

    override fun getAvailableModels(): List<LocalModel> = MODEL_CATALOG

    override fun getFreeSpaceBytes(): Long = getAvailableDiskSpaceBytes(getModelStorageDirectory())

    override fun startDownload(model: LocalModel) {
        cancelDownload()
        downloadJob = scope.launch {
            _downloadingModelId.value = model.id
            _downloadProgress.value = 0f
            _downloadError.value = null
            var tempFile: File? = null
            var notificationStarted = false

            try {
                val modelsDir = getModelStorageDirectory()
                val modelDir = File(modelsDir, model.id)
                modelDir.mkdirs()
                val targetFile = File(modelDir, model.fileName)
                tempFile = File(modelDir, "${model.fileName}.tmp")
                var lastNotifiedPercent = -1

                val freeSpace = getFreeSpaceBytes()
                if (freeSpace < model.sizeBytes + DOWNLOAD_SPACE_BUFFER_BYTES) {
                    _downloadError.value = DownloadError.NOT_ENOUGH_DISK_SPACE
                    return@launch
                }

                @Suppress("DEPRECATION")
                val connection = URL(model.downloadUrl).openConnection() as HttpURLConnection
                connection.instanceFollowRedirects = true
                connection.connectTimeout = 30_000
                connection.readTimeout = 60_000
                connection.connect()

                val responseCode = connection.responseCode
                if (responseCode !in 200..299) {
                    connection.disconnect()
                    throw IOException("Download failed: HTTP $responseCode")
                }

                // Only start the foreground service once we have a live connection.
                // Starting it earlier risks ForegroundServiceDidNotStartInTimeException if
                // the connect() above fails fast (e.g. offline) before the service can run.
                startDownloadNotificationService()
                notificationStarted = true

                val contentLength = connection.contentLengthLong.takeIf { it > 0 } ?: model.sizeBytes
                val buffer = ByteArray(65536)
                var totalBytesRead = 0L

                connection.inputStream.use { input ->
                    tempFile.outputStream().use { output ->
                        while (true) {
                            ensureActive()
                            val bytesRead = input.read(buffer)
                            if (bytesRead <= 0) break
                            output.write(buffer, 0, bytesRead)
                            totalBytesRead += bytesRead
                            val percent = (totalBytesRead * 100 / contentLength).toInt().coerceIn(1, 100)
                            if (percent != lastNotifiedPercent) {
                                lastNotifiedPercent = percent
                                _downloadProgress.value = percent / 100f
                                updateDownloadNotificationProgress(percent)
                            }
                        }
                    }
                }
                connection.disconnect()

                val downloadedSize = tempFile.length()
                if (downloadedSize < contentLength * 0.95) {
                    tempFile.delete()
                    throw IOException("Download incomplete: got $downloadedSize bytes, expected ~$contentLength")
                }

                if (!tempFile.renameTo(targetFile)) {
                    tempFile.copyTo(targetFile, overwrite = true)
                    tempFile.delete()
                }
            } catch (e: Throwable) {
                if (tempFile?.exists() == true) tempFile.delete()
                if (e is CancellationException) throw e
                _downloadError.value = DownloadError.NETWORK_ERROR
            } finally {
                _downloadingModelId.value = null
                _downloadProgress.value = null
                if (notificationStarted) stopDownloadNotificationService()
            }
        }
    }

    override fun cancelDownload() {
        downloadJob?.cancel()
        downloadJob = null
    }

    override suspend fun deleteModel(modelId: String) {
        withContext(Dispatchers.IO) {
            // Wait for any in-flight idle release so its native teardown doesn't race with deleteRecursively().
            idleReleaseJob?.cancelAndJoin()
            idleReleaseJob = null
            if (currentModelId == modelId) {
                release()
            }
            val modelDir = File(getModelStorageDirectory(), modelId)
            modelDir.deleteRecursively()
        }
    }
}
