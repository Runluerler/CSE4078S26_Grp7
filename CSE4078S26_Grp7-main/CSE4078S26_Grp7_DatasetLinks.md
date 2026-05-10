# CSE4078 Spring 2026 - Group 7 Dataset Links

* **Dataset Name:** doktorsitesi
* **Identifier:** alibayram/doktorsitesi
* **Source:** Hugging Face (https://huggingface.co/datasets/alibayram/doktorsitesi)
* **Description:** A comprehensive dataset containing Turkish medical question-answer pairs across various medical branches.
* **Usage in Project:** The raw dataset was filtered to include only the mandatory medical fields specified in the project guidelines (beyin-ve-sinir-cerrahisi, uroloji, vb.). It was then cleaned (deduplicated and null values removed) and split using a stratified approach to maintain class distributions. This resulted in a training set of 38,958 instances for fine-tuning our selected model (Qwen 2.5 - 1.5B) and a strictly unseen test set of 1,500 instances for both baseline and post-fine-tuning evaluations.