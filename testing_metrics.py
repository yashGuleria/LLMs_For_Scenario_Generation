
# Dictionary of airways
airways = {
    "M758": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "A320",
        "departure": {"af": "WMKK"},
        "initial_position": {
            "latitude": "032432.00N",
            "longitude": "1035544.00E",
            "altitude": "FL370",
            "heading": "87.11"
        },
        "air_route": ["IDSEL", "URIGO", "VISAT", "MABAL", "ELGOR", "OPULA", "LUSMO"],
        "destination": {"af": "WBKK"}
    },
    "M758R": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B744",
        "departure": {"af": "WBKK"},
        "initial_position": {
            "latitude": "033341.00N",
            "longitude": "1065534.00E",
            "altitude": "FL340",
            "heading": "270"
        },
        "air_route": ["LUSMO", "OPULA", "ELGOR", "MABAL", "VISAT", "URIGO", "IDSEL", "PADLI", "SAROX"],
        "destination": {"af": "WMKK"}
    },
    "M761": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B738",
        "departure": {"af": "WBGG"},
        "initial_position": {
            "latitude": "020940.00N",
            "longitude": "1075044.00E",
            "altitude": "FL380",
            "heading": "285.33"
        },
        "air_route": ["SABIP", "BOBOB", "OMBAP", "VERIN", "BUNTO", "LIPRO", "KILOT", "OTLON", "KETOD", "OBDAB", "VPK"],
        "destination": {"af": "WMKK"}
    },
    "M761R": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "A333",
        "departure": {"af": "WMKK"},
        "initial_position": {
            "latitude": "032259.00N",
            "longitude": "1032524.00E",
            "altitude": "FL160",
            "heading": "105.36"
        },
        "air_route": ["VPK", "OBDAB", "KETOD", "OTLON", "KILOT", "LIPRO", "BUNTO", "VERIN", "OMBAP", "BOBOB", "SABIP"],
        "destination": {"af": "WBGG"}
    },
    "M771": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B734",
        "departure": {"af": "WSSS"},
        "initial_position": {
            "latitude": "022318.00N",
            "longitude": "1035218.00E",
            "altitude": "FL160",
            "heading": "32.05"
        },
        "air_route": ["VMR", "RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX"],
        "destination": {"af": "ZSFZ"}
    },
    "M771R": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B734",
        "departure": {"af": "ZSFZ"},
        "initial_position": {
            "latitude": "044841.00N",
            "longitude": "1052247.00E",
            "altitude": "FL160",
            "heading": "212.05"
        },
        "air_route": ["DOLOX", "DAMOG", "DUBSA", "VISAT", "OTLON", "RAXIM", "VMR"],
        "destination": {"af": "WSSS"}
    },
    "L635": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B738",
        "departure": {"af": "RCTP"},
        "initial_position": {
            "latitude": "041717.00N",
            "longitude": "1061247.00E",
            "altitude": "FL400",
            "heading": "250"
        },
        "air_route": ["MABLI", "SUSAR", "DUBSA", "UGPEK", "DOVOL", "PADLI"],
        "destination": {"af": "WMKK"}
    },
    "L635R": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B738",
        "departure": {"af": "WMKK"},
        "initial_position": {
            "latitude": "030918.00N",
            "longitude": "1033133.00E",
            "altitude": "FL400",
            "heading": "34.60"
        },
        "air_route": ["PADLI", "DOVOL", "UGPEK", "DUBSA", "SUSAR", "MABLI"],
        "destination": {"af": "RCTP"}
    },
    "N892": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B738",
        "departure": {"af": "RCTP"},
        "initial_position": {
            "latitude": "041717.00N",
            "longitude": "1061247.00E",
            "altitude": "FL400",
            "heading": "230"
        },
        "air_route": ["MABLI", "MUMSO", "MABAL", "KILOT", "KIBOL", "PEKLA", "VMR"],
        "destination": {"af": "WSSS"}
    },
    "N884": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B738",
        "departure": {"af": "WSSS"},
        "initial_position": {
            "latitude": "022318.00N",
            "longitude": "1035218.00E",
            "altitude": "FL400",
            "heading": "65"
        },
        "air_route": ["VMR", "LENDA", "LIPRO", "LEBIN", "ONAPO", "LUSMO"],
        "destination": {"af": "WBKK"}
    },
    "N875": {
        "time": "Initialization time of aircraft after simulation start in seconds",
        "type": "B738",
        "departure": {"af": "VDPP"},
        "initial_position": {
            "latitude": "041225.00N",
            "longitude": "1050014.00E",
            "altitude": "FL400",
            "heading": "132.05"
        },
        "air_route": ["DAMOG", "SUSAR", "MUMSO", "ELGOR", "LEBIN", "OMLIV", "BOBOB"],
        "destination": {"af": "WBGG"}
    }
}


aircraft_types = {'A320', 'B737', 'B738', 'B734', 'B744', 'A388', 'A333'}
