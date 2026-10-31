# Convert the symbols in a MinGW import library into a module-definition file.
#
# mode=import creates the fixed-name import library used by koboldcpp_common.dll.
# mode=forward creates a tiny DLL whose exports forward to target.<symbol>.

BEGIN {
    if (mode == "import") {
        print "LIBRARY koboldcpp_backend.dll"
    } else if (mode == "forward" && target != "") {
        print "LIBRARY koboldcpp_backend.dll"
    } else {
        print "backend_forwarder.awk: expected mode=import or mode=forward with target" > "/dev/stderr"
        exit 1
    }
    print "EXPORTS"
}

function emit_symbol(name, is_data) {
    if (mode == "forward") {
        printf "    %s=%s.%s", name, target, name
    } else {
        printf "    %s", name
    }
    if (is_data) {
        printf " DATA"
    }
    printf "\n"
    symbol_count++
}

# Function imports have a normal text thunk in the import library.
$2 == "T" {
    emit_symbol($3, 0)
}

# Data imports have __imp_<name> and __nm_<name> records but no text thunk.
# Use the latter so each data export is emitted exactly once.
$2 == "I" && $3 ~ /^__nm_/ {
    name = $3
    sub(/^__nm_/, "", name)
    emit_symbol(name, 1)
}

END {
    if (symbol_count == 0) {
        print "backend_forwarder.awk: no exported symbols found" > "/dev/stderr"
        exit 1
    }
}
