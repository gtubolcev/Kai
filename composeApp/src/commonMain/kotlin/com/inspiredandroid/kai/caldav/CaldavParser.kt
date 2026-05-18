package com.inspiredandroid.kai.caldav

object CaldavParser {

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
