# Download training set
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B6eKvaijfFUDQUUwd21EckhUbWs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B6eKvaijfFUDQUUwd21EckhUbWs" -O trainset.zip && rm -rf /tmp/cookies.txt

# Download validation set
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B6eKvaijfFUDd3dIRmpvSk8tLUk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B6eKvaijfFUDd3dIRmpvSk8tLUk" -O valset.zip && rm -rf /tmp/cookies.txt

# Download test set
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B6eKvaijfFUDbW4tdGpaYjgzZkU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B6eKvaijfFUDbW4tdGpaYjgzZkU" -O testset.zip && rm -rf /tmp/cookies.txt

# Make dataset directory
mkdir ./WIDER

# Unzip the downloaded zip file
unzip ./trainset.zip -d ./WIDER
unzip ./valset.zip -d ./WIDER
unzip ./testset.zip -d ./WIDER

mv ./WIDER/WIDER_train ./WIDER/train
mv ./WIDER/WIDER_val ./WIDER/val
mv ./WIDER/WIDER_test ./WIDER/test

# Remove the downloaded zip file
rm ./trainset.zip
rm ./valset.zip
rm ./testset.zip