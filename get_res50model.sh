# Download dataset from Dropbox
if ! [ -f "pretrain/resnet50_ft_weight.pkl" ]; then
    wget https://www.dropbox.com/s/6tv01ngob44ihsr/resnet50_ft_weight.zip
fi

# Unzip the downloaded zip file
unzip ./resnet50_ft_weight.zip

# Remove the downloaded zip file
rm ./resnet50_ft_weight.zip

# move to target folder
mkdir pretrain
mv resnet50_ft_weight.pkl ./pretrain/