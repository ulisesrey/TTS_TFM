"""
This filke as it is now only downloads the first 100 samples
# TODO The script should be modified to iterate over offset and length to download all samples
# TODO The number of rows is in the json file! num_rows_total
"""
import urllib.parse
import requests



def download_json(user, dataset, config, split, offset, length, output):
    # URL-encode the dataset name (user/dataset)
    # Converts / to %2F
    encoded_dataset = urllib.parse.quote(f"{user}/{dataset}")

    # Construct the URL
    url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={encoded_dataset}"
        f"&config={config}"
        f"&split={split}"
        f"&offset={offset}"
        f"&length={length}"
    )

    # Fetch and save the data
    print(f"Downloading from: {url}")
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if request failed

    with open(output, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Saved to {output}")


if __name__ == "__main__":
    
    # CONFIG VARIABLES
    user = "ylacombe"
    datasets = ["google-colombian-spanish",
            "google-argentinian-spanish",
            "google-chilean-spanish"]
    configs = ["female", "male"]
    split = "train"
    offset = 0
    length = 100
    # output name is defined withthin the loop
    
    # Loop through datasets and configs
    for dataset in datasets:
        for config in configs:
            output = f"data/dataset_json/{user}_{dataset}_{config}_{offset}_{length}.json"
            download_json(user, dataset, config, split, offset, length, output)