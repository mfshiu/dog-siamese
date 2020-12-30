class Config:
    training_dir = "./data/dogs/training/"
    testing_dir = "./data/dogs/testing/"
    train_batch_size = 64
    train_number_epochs = 100

    class Evaluate:
        model_path = "trained/DogSiamese-gpu.pkl"
        inference_output_path = "output/inferences.tsv"
        eer_output_path = "output/eer.tsv"
        dog_input_root = "data/verification"
        # dog_input_root = "data/dogs/testing"
        dog_count = 85
        group_size = 2


class FaceConfig:
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 100

    class Evaluate:
        model_path = "trained/DogSiamese-gpu.pkl"
        inference_output_path = "output/inferences.tsv"
        eer_output_path = "output/eer.tsv"
        dog_input_root = "data/faces/testing"
        dog_count = 3
        group_size = 10
        inference_output_path
