#ifndef args_h
#define args_h

#include <stdio.h>
#include <iostream>
#include <vector>

struct arg {
    std::string full_name;
    std::string abbreviation = "";
    std::string description = "";
    bool required = false;
    bool has_param = false;

    std::string help_text();
};

struct bool_arg : public arg {
    bool_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, bool val = false) {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };

    bool value = false;
};

struct string_arg : public arg {
    string_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, std::string val = "") {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };
    bool has_param = true;
    std::string value;

    int parse(int argc, const char ** argv);
};

struct int_arg : public arg {
    int_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, int * val = nullptr) {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };

    int * value;

    int parse(int argc, const char ** argv);

};

struct float_arg : public arg {
    float_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, float * val = nullptr) {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };

    bool has_param = true;
    float * value;

    int parse(int argc, const char ** argv);
};

struct arg_list {
    std::vector<float_arg> fargs;
    std::vector<int_arg> iargs;
    std::vector<bool_arg> bargs;
    std::vector<string_arg> sargs;
    bool for_help = false;

    void add_argument(float_arg arg) {
        fargs.push_back(arg);
    }

    void add_argument(int_arg arg) {
        iargs.push_back(arg);
    }

    void add_argument(bool_arg arg) {
        bargs.push_back(arg);
    }

    void add_argument(string_arg arg) {
        sargs.push_back(arg);
    }

    void help();

    void validate();

    void parse(int argc, const char ** argv);

    int find_and_parse(std::string name, int argc, const char ** argv);

    std::string get_string_param(std::string full_name);

    int * get_int_param(std::string full_name);

    float * get_float_param(std::string full_name);

    bool get_bool_param(std::string full_name);
};

#endif

