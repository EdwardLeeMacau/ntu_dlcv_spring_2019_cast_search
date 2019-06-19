# Download dataset from Dropbox
wget https://www.dropbox.com/s/6tv01ngob44ihsr/resnet50_ft_weight.zip

# Unzip the downloaded zip file
unzip ./resnet50_gt_weight.zip

# Remove the downloaded zip file
rm ./resnet50_gt_weight.zip

# move to target folder
mkdir pretrain
mv resnet50_gt_weight.pkl ./pretrain/