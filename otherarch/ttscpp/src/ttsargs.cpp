#include "ttsargs.h"

std::string arg::help_text() {
    std::string htxt = full_name;
    if (abbreviation != "") {
        htxt += " (" + abbreviation + ")";
    }
    htxt += ":\n    ";
    if (description != "") {
        htxt += description + "\n";
    } else {
        htxt += "is a " + (std::string)(required ? "required " : "optional ") + "parameter.\n";
    }
    return htxt;
}

int string_arg::parse(int argc, const char ** argv) {
    required = false;
    value.assign(argv[0]);
    return 1;
}

int int_arg::parse(int argc, const char ** argv) {
    if (required) {
        required = false;
    }
    int val = atoi(argv[0]);
    *value = val;
    return 1;
}

int float_arg::parse(int argc, const char ** argv) {
    if (required) {
        required = false;
    }
    float val = strtof(argv[0], nullptr);
    *value = val;
    return 1;
}

void arg_list::help() {
    std::string help_text = "";
    for (auto arg : fargs) {
        help_text += arg.help_text();
    }
    for (auto arg : iargs) {
        help_text += arg.help_text();

    }
    for (auto arg : bargs) {
        help_text += arg.help_text();

    }
    for (auto arg : sargs) {
        help_text += arg.help_text();

    }
    fprintf(stdout, "%s", help_text.c_str());
}

void arg_list::validate() {
    for (auto arg : fargs) {
        if (arg.required) {
            fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
            exit(1);
        }
    }
    for (auto arg : iargs) {
        if (arg.required) {
            fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
            exit(1);
        }
    }
    for (auto arg : bargs) {
        if (arg.required) {
            fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
            exit(1);
        }
    }
    for (auto arg : sargs) {
        if (arg.required) {
            fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
            exit(1);
        }
    }
}

void arg_list::parse(int argc, const char ** argv) {
    int current_arg = 1;
    while (current_arg < argc) {
        std::string name(argv[current_arg]);
        if (name == "--help") {
            for_help = true;
            return;
        }
        current_arg += 1;
        current_arg += find_and_parse(name, argc - current_arg, argv + current_arg);
    }
}

int arg_list::find_and_parse(std::string name, int argc, const char ** argv) {
    for (int i = 0; i < fargs.size(); i++) {
        if (fargs[i].full_name == name || fargs[i].abbreviation == name) {
            return fargs[i].parse(argc, argv);
        }
    }
    for (int i = 0; i < iargs.size(); i++) {
        if (iargs[i].full_name == name || iargs[i].abbreviation == name) {
            return iargs[i].parse(argc, argv);
        }
    }
    for (int i = 0; i < bargs.size(); i++) {
        if (bargs[i].full_name == name || bargs[i].abbreviation == name) {
            bargs[i].value = !bargs[i].value;
            bargs[i].required = false;
            return 0;
        }

    }
    for (int i = 0; i < sargs.size(); i++) {
        if (sargs[i].full_name == name || sargs[i].abbreviation == name) {
            return sargs[i].parse(argc, argv);
        }
    }
    fprintf(stderr, "argument '%s' is not a valid argument. Call '--help' for information on all valid arguments.\n", name.c_str());
    exit(1);
}

std::string arg_list::get_string_param(std::string full_name) {
    for (auto arg : sargs) {
        if (arg.full_name == full_name) {
            return arg.value;
        }
    }
    return "";
}

int * arg_list::get_int_param(std::string full_name) {
    for (auto arg : iargs) {
        if (arg.full_name == full_name) {
            return arg.value;
        }
    }
    return nullptr;
}

float * arg_list::get_float_param(std::string full_name) {
    for (auto arg : fargs) {
        if (arg.full_name == full_name) {
            return arg.value;
        }
    }
    return nullptr;
}

bool arg_list::get_bool_param(std::string full_name) {
    for (auto arg : bargs) {
        if (arg.full_name == full_name) {
            return arg.value;
        }
    }
    return false;
}

