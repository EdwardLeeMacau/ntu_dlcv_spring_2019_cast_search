# Make resize dataset directory
if ! [ -d "./IMDb_resize" ]; then
    mkdir IMDb_resize    
fi

# Get resize test dataset
if ! [ -d "./IMDb_resize/test" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1641vXY5DtlVWeKol6M4PtVkfdCnN7328' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1641vXY5DtlVWeKol6M4PtVkfdCnN7328" -O "./IMDb_resize/test_resize.zip" && rm -rf /tmp/cookies.txt
fi

# Get resize val dataset
if ! [ -d "./IMDb_resize/val" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1zGVd42yeKgs4Hl-15yM7VUSXqY5BYR39' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zGVd42yeKgs4Hl-15yM7VUSXqY5BYR39" -O "./IMDb_resize/Resize_val.tar.gz" && rm -rf /tmp/cookies.txt
fi

# Get resize train dataset
if ! [ -d "./IMDb_resize/train" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=168wyQ2EgwnuTVCwzAkglRnIvY4vQWKxj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=168wyQ2EgwnuTVCwzAkglRnIvY4vQWKxj" -O "./IMDb_resize/Resize_train.tar.gz" && rm -rf /tmp/cookies.txt
fi

# Unzip and remove the downloaded zip file
unzip "IMDb_resize/train.zip" -d ./IMDb_resize
tar -xvzf "IMDb_resize/Resize_val.tar.gz" -C ./IMDb_resize
tar -xvzf "IMDb_resize/Resize_train.tar.gz" -C ./IMDb_resize

rm ./IMDb_resize/Resize_train.tar.gz
rm ./IMDb_resize/Resize_val.tar.gz
rm ./IMDb_resize/test.zip