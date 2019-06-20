# get test_resize.zip
file_id="1641vXY5DtlVWeKol6M4PtVkfdCnN7328"
file_name="test_resize.zip"
    
# first stage to get the warning html
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$file_id" > /tmp/intermezzo.html

# second stage to extract the download link from html above
download_link=$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')
curl -L -b /tmp/cookies "https://drive.google.com$download_link" > $file_name


# get Resize_val.tar.gz
file_id="1zGVd42yeKgs4Hl-15yM7VUSXqY5BYR39"
file_name="Resize_val.tar.gz"
    
# first stage to get the warning html
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$file_id" > /tmp/intermezzo.html

# second stage to extract the download link from html above
download_link=$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')
curl -L -b /tmp/cookies "https://drive.google.com$download_link" > $file_name