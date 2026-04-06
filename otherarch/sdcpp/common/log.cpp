#include "log.h"

bool log_verbose = false;
bool log_color   = false;

std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

void print_utf8(FILE* stream, const char* utf8) {
    if (!utf8) {
        return;
    }

#ifdef _WIN32
    HANDLE h = (stream == stderr)
                   ? GetStdHandle(STD_ERROR_HANDLE)
                   : GetStdHandle(STD_OUTPUT_HANDLE);

    DWORD mode;
    BOOL is_console = GetConsoleMode(h, &mode);

    if (is_console) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
        if (wlen <= 0) {
            return;
        }

        wchar_t* wbuf = (wchar_t*)malloc(wlen * sizeof(wchar_t));
        if (!wbuf) {
            return;
        }

        MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wbuf, wlen);

        DWORD written;
        WriteConsoleW(h, wbuf, wlen - 1, &written, NULL);

        free(wbuf);
    } else {
        DWORD written;
        WriteFile(h, utf8, (DWORD)strlen(utf8), &written, NULL);
    }
#else
    fputs(utf8, stream);
#endif
}

void log_print(enum sd_log_level_t level, const char* log, bool verbose, bool color) {
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default:
            tag_color = 33;
            level_str = "?????";
            break;
    }

    if (color) {
        fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
    } else {
        fprintf(out_stream, "[%-5s] ", level_str);
    }
    print_utf8(out_stream, log);
    fflush(out_stream);
}

void example_log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...) {
    constexpr size_t LOG_BUFFER_SIZE = 4096;

    va_list args;
    va_start(args, format);

    static char log_buffer[LOG_BUFFER_SIZE + 1];
    int written = snprintf(log_buffer, LOG_BUFFER_SIZE, "%s:%-4d - ", sd_basename(file).c_str(), line);

    if (written >= 0 && written < static_cast<int>(LOG_BUFFER_SIZE)) {
        vsnprintf(log_buffer + written, LOG_BUFFER_SIZE - written, format, args);
    }
    size_t len = strlen(log_buffer);
    if (len == 0 || log_buffer[len - 1] != '\n') {
        strncat(log_buffer, "\n", LOG_BUFFER_SIZE - len);
    }

    log_print(level, log_buffer, log_verbose, log_color);

    va_end(args);
}
