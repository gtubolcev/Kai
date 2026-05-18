@file:OptIn(kotlin.io.encoding.ExperimentalEncodingApi::class)

package com.inspiredandroid.kai.caldav

import com.inspiredandroid.kai.httpClient
import io.ktor.client.request.delete
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.put
import io.ktor.client.request.request
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpMethod
import io.ktor.http.isSuccess
import kotlin.io.encoding.Base64

class CaldavClient(
    private val username: String,
    private val password: String,
) {
    private val client = httpClient {}

    private fun basicAuth(): String =
        "Basic ${Base64.encode("$username:$password".encodeToByteArray())}"

    suspend fun propfind(url: String): Result<Unit> = runCatching {
        val response = client.request(url) {
            method = HttpMethod("PROPFIND")
            header("Authorization", basicAuth())
            header("Depth", "0")
            header("Content-Type", "application/xml")
            setBody("""<?xml version="1.0" encoding="UTF-8"?><D:propfind xmlns:D="DAV:"><D:prop><D:resourcetype/></D:prop></D:propfind>""")
        }
        when (response.status.value) {
            401 -> throw CaldavException("Invalid credentials")
            404 -> throw CaldavException("Collection not found")
            else -> if (!response.status.isSuccess() && response.status.value != 207) {
                throw CaldavException("HTTP ${response.status.value}: ${response.bodyAsText()}")
            }
        }
    }

    suspend fun put(url: String, icsBody: String): Result<Unit> = runCatching {
        val response = client.put(url) {
            header("Authorization", basicAuth())
            header("Content-Type", "text/calendar; charset=utf-8")
            setBody(icsBody)
        }
        when (response.status.value) {
            401 -> throw CaldavException("Invalid credentials")
            404 -> throw CaldavException("Collection not found")
            else -> if (!response.status.isSuccess()) {
                throw CaldavException("HTTP ${response.status.value}: ${response.bodyAsText()}")
            }
        }
    }

    suspend fun get(url: String): Result<String> = runCatching {
        val response = client.get(url) {
            header("Authorization", basicAuth())
        }
        when (response.status.value) {
            401 -> throw CaldavException("Invalid credentials")
            404 -> throw CaldavException("Not found")
            else -> if (!response.status.isSuccess()) {
                throw CaldavException("HTTP ${response.status.value}")
            }
        }
        response.bodyAsText()
    }

    suspend fun report(url: String, xmlBody: String): Result<String> = runCatching {
        val response = client.request(url) {
            method = HttpMethod("REPORT")
            header("Authorization", basicAuth())
            header("Depth", "1")
            header("Content-Type", "application/xml")
            setBody(xmlBody)
        }
        when (response.status.value) {
            401 -> throw CaldavException("Invalid credentials")
            404 -> throw CaldavException("Collection not found")
            else -> if (!response.status.isSuccess() && response.status.value != 207) {
                throw CaldavException("HTTP ${response.status.value}: ${response.bodyAsText()}")
            }
        }
        response.bodyAsText()
    }

    suspend fun delete(url: String): Result<Unit> = runCatching {
        val response = client.delete(url) {
            header("Authorization", basicAuth())
        }
        when (response.status.value) {
            401 -> throw CaldavException("Invalid credentials")
            404 -> throw CaldavException("Not found")
            else -> if (!response.status.isSuccess()) {
                throw CaldavException("HTTP ${response.status.value}: ${response.bodyAsText()}")
            }
        }
    }
}

class CaldavException(message: String) : Exception(message)
