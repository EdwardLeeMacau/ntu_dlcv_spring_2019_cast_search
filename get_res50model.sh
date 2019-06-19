# Make dataset directory
if ! [ -d "pretrain" ]; then
    mkdir pretrain
fi

# Download dataset from Dropbox
if ! [ -f "pretrain/resnet50_ft_weight.pkl" ]; then
    wget https://www.dropbox.com/s/6tv01ngob44ihsr/resnet50_ft_weight.zip
fi

# Unzip the downloaded zip file
unzip ./resnet50_ft_weight.zip

# Remove the downloaded zip file
rm ./resnet50_ft_weight.zip

# move to target folder
mv resnet50_ft_weight.pkl ./pretrain/