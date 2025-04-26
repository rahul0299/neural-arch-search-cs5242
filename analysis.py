import pandas as pd
import Log_metrics_extract as lm
import os
import matplotlib.pyplot as plt
import random


def get_full_training_log_data(experiment_name):
    log_file_dir = os.path.join("LogFiles", "TrainingLogs")
    log_output_dir = os.path.join("LogFiles", "Output")
    os.makedirs(log_file_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    log_file_path = os.path.join(log_file_dir, f"{experiment_name}.log")
    log_output_path = os.path.join(log_output_dir, f"{experiment_name}_log.csv")

    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file does not exist at : {log_file_path}")

    lm.parse_training_log(log_file_path, log_output_path)

    df_train_log = pd.read_csv(os.path.join("Results", "Controller", "CNN", f"{experiment_name}.csv"))
    df_stats = pd.read_csv(log_output_path)

    return pd.concat((df_train_log, df_stats), axis=1)


def plot_token_distribution(df, experiment_name, include_layer1=True, use_sns=False, sns_palette="tab20c"):
    token_map = {
        "channels": ["layer_1_channels", "layer_2_channels", "layer_3_channels"],
        "filter": ["layer_1_filter", "layer_2_filter", "layer_3_filter"],
        "padding": ["layer_1_padding", "layer_2_padding", "layer_3_padding"]
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    fig.suptitle(f"{experiment_name} Token Distribution", fontsize=16)

    for i, token_type in enumerate(["channels", "filter", "padding"]):
        layers = token_map[token_type]
        if not include_layer1:
            layers = layers[1:]

        df_tokens = df[layers].copy()
        df_tokens.columns = [f"Layer {j+1 if include_layer1 else j+2}" for j in range(len(layers))]
        df_melted = df_tokens.melt(var_name="Layer", value_name="Token")

        token_counts = df_melted.groupby(["Layer", "Token"]).size().reset_index(name="Count")
        pivot = token_counts.pivot(index="Layer", columns="Token", values="Count").fillna(0)

        ax = axs[i]
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap=sns_palette if use_sns else None)
        ax.set_title(f"{token_type.capitalize()} Token Distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Layer")
        ax.set_xticklabels(pivot.index, rotation=0)
        ax.legend(title=token_type.capitalize(), bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis='y')

    plt.tight_layout()
    plt.show()


def compute_stats(df, step=50, metrics=None, stats=None,limit=None):
    if metrics is None:
        metrics = df.columns.tolist()
    if stats is None:
        stats = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    if limit is not None:
        df = df.head(limit)

    df = df.copy()
    df["row_index"] = df.index
    df["range_label"] = df["row_index"].apply(lambda x: f"{(x // step) * step}-{(x // step + 1) * step - 1}")

    grouped = df.groupby("range_label", sort=False)[metrics].describe()

    if isinstance(grouped.columns, pd.MultiIndex):
        grouped = grouped.loc[:, grouped.columns.get_level_values(1).isin(stats)]

    return grouped


def get_random_color_hex_code():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def plot_metrics(df, experiment_name, metrics=None, labels=None, titles=None, colors=None, limit=None, references=None):
    if limit is not None:
        df = df.head(limit)

    if metrics is None:
        metrics = df.columns
    else:
        for m in metrics:
            if m not in df.columns:
                raise ValueError(f"Metric '{m}' not found in progress data.")

    if labels is None:
        labels = { m: m for m in metrics }

    if titles is None:
        titles = { m: m.replace("_", " ").capitalize() for m in metrics }

    if colors is None:
        colors = {m: get_random_color_hex_code() for m in metrics}

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

    # fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"{experiment_name} Controller Training Progress", fontsize=16)

    for i, metric in enumerate(metrics):
        axs[i].plot(df[metric], label=labels[metric], color=colors[metric] if metric in colors else get_random_color_hex_code())


        # SAMPLE REFERENCE
        # references = {
        #     "test_acc": [
        #         {
        #             "value": 0.7765,
        #             "label": "LeNet",
        #             "color": "tab:blue"
        #         },
        #         {
        #             "value": 0.8125,
        #             "label": "VGG11"
        #         }
        #     ]
        # }


        if references is not None and metric in references:
            for ref in references[metric]:
                axs[i].axhline(ref["value"], color=ref["color"] if "color" in ref else get_random_color_hex_code(), linestyle='--', label=ref["label"])

        axs[i].set_title(titles[metric])
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel(metric)
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compare_metrics(df_list, df_names, df_colors, metrics=None, labels=None, titles=None, limit=None):
    if limit is not None:
        for i, df in enumerate(df_list):
            df_list[i] = df.head(limit)

    if metrics is None:
        metrics = df_list[0].columns
    else:
        for m in metrics:
            for df in df_list:
                if m not in df.columns:
                    raise ValueError(f"Metric '{m}' not found in progress data.")

    if labels is None:
        labels = { m: m for m in metrics }

    if titles is None:
        titles = { m: m.replace("_", " ").capitalize() for m in metrics }

    fig, axs = plt.subplots(len(metrics), 1, figsize=(15, 4 * len(metrics)))

    # fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(" vs ".join(df_names), fontsize=16)

    for i, metric in enumerate(metrics):
        for j, df in enumerate(df_list):
            axs[i].plot(df[metric], label=labels[metric], color=df_colors[j])

        axs[i].set_title(titles[metric])
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel(metric)
        axs[i].grid(True)
        axs[i].legend(df_names)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
