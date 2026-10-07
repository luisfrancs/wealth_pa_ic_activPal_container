from pathlib import Path

from wealth_pb_ee_models.models.pretrained_models import multi_taks_7classes_AP
from wealth_pb_ee_models.utils.utils import predict_single_file_AP

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    input_files = sorted(INPUT_DIR.glob("*.datx"))

    if not input_files:
        raise RuntimeError(
            "No DATX files found in /app/input"
        )

    print(f"Found {len(input_files)} input files.")

    for input_file in input_files:

        print(f"Processing: {input_file.name}")

        # WEALTH activPAL inference goes here
        #
        # data = load_activpal(input_file)
        # data = preprocess(data)
        # predictions = predict(data)
        # save_predictions(...)
        ####
        loaded_model, encoding_dict_PB, encoding_dict_EE =multi_taks_7classes_AP()
        file_name=str(input_file) #transform the path to string
        #Predic PB and EE (it can load activPAL files: .datx (compressed) files and uncompresed .csv files)
        df_predicted, _ = predict_single_file_AP(
            file_name, #Path to activPAL data file
            loaded_model, #Load trained model
            encoding_dict_PB, #Load Dictionnary for PB
            encoding_dict_EE) #Load Dictionnary for EE
        ####
        output_file = (
            OUTPUT_DIR /
            f"{input_file.stem}_predictions.csv"
        )

        df_predicted.to_csv(output_file, index=False)

        print(f"Saved predictions to: {output_file}")


if __name__ == "__main__":
    main()