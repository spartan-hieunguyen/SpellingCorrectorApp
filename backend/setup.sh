python -m pip install -e .

cd autocorrection
mkdir input
mkdir input/luanvan
cd input/luanvan
gdown https://drive.google.com/uc?id=145geEupadzGwxZaueZE-kr4fHYmO-REC
cd ../..

mkdir weights
mkdir weights/history
mkdir weights/model
gdown https://drive.google.com/uc?id=1PqGIjiQCp5xNINsAcBzOracLw0t6CuVA
cd ../..

cd tokenization_repair
mkdir data/estimators/
mkdir data/estimators/lm/
cd data/estimators
gdown https://drive.google.com/drive/folders/1zhtQmPTah7qneEHPHuFHuf3qd5R047-W -O / --folder
cd ../lm
gdown https://drive.google.com/drive/folders/1lG3swcUyUPYOf4ziJGfPQboaYtD5SgTM -O / --folder