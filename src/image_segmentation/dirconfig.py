import os
import subprocess
import shutil

# Função para rodar comandos no shell
def run_command(command):
    subprocess.run(command, shell=True, check=True)

if __name__ == '__main__':
    # Criar diretórios
    os.makedirs("Image_Segmentation", exist_ok=True)

    # Baixar os arquivos com wget
    urls = [
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip"
    ]

    for url in urls:
        run_command(f"wget {url}")

    # Descompactar os arquivos zip
    zip_files = [
        "ISIC2018_Task1-2_Training_Input.zip",
        "ISIC2018_Task1_Training_GroundTruth.zip",
        "ISIC2018_Task1-2_Validation_Input.zip",
        "ISIC2018_Task1_Validation_GroundTruth.zip",
        "ISIC2018_Task1-2_Test_Input.zip",
        "ISIC2018_Task1_Test_GroundTruth.zip"
    ]

    for zip_file in zip_files:
        run_command(f"unzip {zip_file}")

    # Criar diretório dataset
    os.makedirs("dataset", exist_ok=True)

    # Criar subdiretórios dentro de dataset
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/train_GT", exist_ok=True)
    os.makedirs("dataset/valid", exist_ok=True)
    os.makedirs("dataset/valid_GT", exist_ok=True)
    os.makedirs("dataset/test", exist_ok=True)
    os.makedirs("dataset/test_GT", exist_ok=True)

    # Mover arquivos para seus respectivos diretórios
    shutil.move("ISIC2018_Task1-2_Training_Input", "dataset/train")
    shutil.move("ISIC2018_Task1_Training_GroundTruth", "dataset/train_GT")
    shutil.move("ISIC2018_Task1-2_Validation_Input", "dataset/valid")
    shutil.move("ISIC2018_Task1_Validation_GroundTruth", "dataset/valid_GT")
    shutil.move("ISIC2018_Task1-2_Test_Input", "dataset/test")
    shutil.move("ISIC2018_Task1_Test_GroundTruth", "dataset/test_GT")

    # Remover os arquivos .zip
    for zip_file in zip_files:
        os.remove(zip_file)

    # Excluir arquivos não desejados (não .jpg ou .png) dentro de dataset
    for root, dirs, files in os.walk("dataset"):
        for file in files:
            if not (file.endswith(".jpg") or file.endswith(".png")):
                os.remove(os.path.join(root, file))

    print("Processamento concluído com sucesso!")
