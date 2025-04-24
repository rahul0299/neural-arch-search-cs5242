import re

def extract_conv_params_from_log(file_path):
    results = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    inside_model = False
    conv_layers = []
    model_index = -1

    for line in lines:
        line = line.strip()

        # Detect start of a new model block
        if "ChildCNNModel(" in line:
            inside_model = True
            conv_layers = []
            model_index += 1
            continue

        if inside_model:
            # End of model block
            if line == ")":
                inside_model = False
                if len(conv_layers) >= 3:
                    # Extract parameters from 2nd and 3rd Conv2d layers
                    def count_params(in_ch, out_ch, kh, kw, bias=True):
                        return in_ch * out_ch * kh * kw + (out_ch if bias else 0)

                    in2, out2, kh2, kw2 = conv_layers[1]
                    in3, out3, kh3, kw3 = conv_layers[2]

                    conv2_params = count_params(in2, out2, kh2, kw2)
                    conv3_params = count_params(in3, out3, kh3, kw3)

                    results.append({
                        'iteration': model_index,
                        'conv2_params': conv2_params,
                        'conv3_params': conv3_params,
                        'total_params': conv2_params + conv3_params
                    })
                continue

            # Parse Conv2d lines regardless of prefix
            if "Conv2d" in line:
                match = re.search(r"Conv2d\((\d+), (\d+), kernel_size=\((\d+), (\d+)\)", line)
                if match:
                    conv_layers.append(tuple(map(int, match.groups())))

    return results

# Run the script
results = extract_conv_params_from_log("Sample_Log.log")

# Print results
for r in results:
    print(f"Iteration {r['iteration']}: Conv2 Params = {r['conv2_params']}, "
          f"Conv3 Params = {r['conv3_params']}, Total = {r['total_params']}")

import csv

# Save results to CSV
with open("conv2_conv3_params.csv", mode="w", newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["iteration", "conv2_params", "conv3_params", "total_params"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Exported to conv2_conv3_params.csv")
