package com.inspiredandroid.kai.caldav

object CaldavParser {

    // Replaces or adds iCalendar properties within a single component (VEVENT/VTODO).
    // Strips line folding first so property keys are reliably matched, then re-joins
    // with CRLF as required by RFC 5545.
    fun updateProperties(ics: String, updates: Map<String, String>): String {
        val unfolded = ics
            .replace("\r\n ", "").replace("\r\n\t", "")
            .replace("\n ", "").replace("\n\t", "")
        val lines = unfolded.split(Regex("\r\n|\r|\n")).filter { it.isNotBlank() }.toMutableList()
        val handled = mutableSetOf<String>()

        val result = lines.map { line ->
            val colonIdx = line.indexOf(':')
            if (colonIdx > 0) {
                val baseKey = line.substring(0, colonIdx).substringBefore(';').uppercase()
                if (baseKey in updates) {
                    handled.add(baseKey)
                    "$baseKey:${updates[baseKey]}"
                } else line
            } else line
        }.toMutableList()

        // Insert properties that weren't already present, before END:VEVENT or END:VTODO
        val toInsert = updates.keys.map { it.uppercase() }.filter { it !in handled }
        if (toInsert.isNotEmpty()) {
            val insertAt = result.indexOfFirst {
                it.equals("END:VEVENT", ignoreCase = true) || it.equals("END:VTODO", ignoreCase = true)
            }
            if (insertAt >= 0) toInsert.forEach { key -> result.add(insertAt, "$key:${updates[key]}") }
        }

        return result.joinToString("\r\n")
    }

    fun parseEvents(xml: String): List<Map<String, String>> =
        extractCalendarDataBlocks(xml).mapNotNull { parseComponent(it, "VEVENT") }

    fun parseTasks(xml: String): List<Map<String, String>> =
        extractCalendarDataBlocks(xml).mapNotNull { parseComponent(it, "VTODO") }

    private fun extractCalendarDataBlocks(xml: String): List<String> {
        val blocks = mutableListOf<String>()
        // Match <ns:calendar-data> regardless of namespace prefix
        val regex = Regex("""<[^:>\s]+:calendar-data[^>]*>([\s\S]*?)</[^:>\s]+:calendar-data>""")
        regex.findAll(xml).forEach { match ->
            val content = match.groupValues[1].trim()
            if (content.isNotBlank()) blocks.add(content)
        }
        return blocks
    }

    // RFC 5545 section 3.1: unfold CRLF+SPACE and LF+SPACE continuations, then split into lines.
    private fun unfoldLines(ics: String): List<String> =
        ics
            .replace("\r\n ", "").replace("\r\n\t", "")
            .replace("\n ", "").replace("\n\t", "")
            .split(Regex("\r\n|\r|\n"))
            .filter { it.isNotBlank() }

    private fun parseComponent(ics: String, componentType: String): Map<String, String>? {
        val lines = unfoldLines(ics)
        val props = mutableMapOf<String, String>()
        var inComponent = false

        for (line in lines) {
            when {
                line == "BEGIN:$componentType" -> {
                    inComponent = true
                    props.clear()
                }
                line == "END:$componentType" && inComponent -> return props.ifEmpty { null }
                inComponent -> {
                    val colonIdx = line.indexOf(':')
                    if (colonIdx > 0) {
                        // Strip property parameters: DTSTART;TZID=Europe/Berlin → DTSTART
                        val baseKey = line.substring(0, colonIdx).substringBefore(';').uppercase()
                        val value = line.substring(colonIdx + 1)
                        props[baseKey] = value
                    }
                }
            }
        }
        return null
    }
}
