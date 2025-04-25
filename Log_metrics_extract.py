import re
import csv

# === CONFIGURATION ===
def parse_training_log(LOG_PATH="Sample_Log.log", CSV_OUTPUT_PATH="final_conv_layer_params.csv"):
    def count_conv2d_params(in_ch, out_ch, kh, kw, bias=True):
        return in_ch * out_ch * kh * kw + (out_ch if bias else 0)

    results = []
    inside_model = False
    iteration = -1
    conv_layers = []

    with open(LOG_PATH, 'r') as f:
        for line in f:
            line = line.strip()

            if "------ ITERATION" in line:
                iteration = int(re.search(r"ITERATION (\d+)", line).group(1))

            if "ChildCNNModel(" in line:
                inside_model = True
                conv_layers = []
                continue

            if inside_model:
                if "Conv2d" in line:
                    # Match with optional padding
                    match = re.search(
                        r"Conv2d\((\d+), (\d+), kernel_size=\((\d+), (\d+)\)(?:, stride=\([^)]+\))?(?:, padding=\((\d+), (\d+)\))?",
                        line
                    )
                    if match:
                        in_ch, out_ch, kh, kw, pad_h, pad_w = match.groups()
                        pad_h = int(pad_h) if pad_h is not None else 0
                        pad_w = int(pad_w) if pad_w is not None else 0
                        conv_layers.append((int(in_ch), int(out_ch), int(kh), int(kw), pad_h))

                elif line == ")":
                    inside_model = False
                    if len(conv_layers) >= 3:
                        row = {"iteration": iteration}
                        total = 0
                        for i in range(3):
                            in_c, out_c, kh, kw, pad = conv_layers[i]
                            param_count = count_conv2d_params(in_c, out_c, kh, kw)
                            row[f"layer_{i + 1}_channels"] = out_c
                            row[f"layer_{i + 1}_filter"] = kh
                            row[f"layer_{i + 1}_padding"] = pad
                            row[f"layer_{i + 1}_param_count"] = param_count
                            total += param_count
                        row["total_param_count"] = total
                        results.append(row)

    # CSV Output
    fieldnames = [
        "iteration",
        "layer_1_channels", "layer_1_filter", "layer_1_padding", "layer_1_param_count",
        "layer_2_channels", "layer_2_filter", "layer_2_padding", "layer_2_param_count",
        "layer_3_channels", "layer_3_filter", "layer_3_padding", "layer_3_param_count",
        "total_param_count"
    ]

    with open(CSV_OUTPUT_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Fixed padding captured and exported {len(results)} models to {CSV_OUTPUT_PATH}")
