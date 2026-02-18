import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "naderabdalghani/iam-handwritten-forms-dataset"
)

print("Path to dataset files:", path)
