def override_yaml(yaml_config, to_override):
    if to_override is not None:
        for new_setting in to_override:
            if new_setting.count(":") == 1:
                key, value = new_setting.split(":")
                value_type_func = type(yaml_config[key])
                if value_type_func == bool:
                    yaml_config[key] = value == "true"
                else:
                    yaml_config[key] = value_type_func(value)

            elif new_setting.count(":") == 2:
                conf, key, value = new_setting.split(":")
                value_type_func = type(yaml_config[conf][key])
                if value_type_func == bool:
                    yaml_config[conf][key] = value == "true"
                else:
                    yaml_config[conf][key] = value_type_func(value)

    return yaml_config
