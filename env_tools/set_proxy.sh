PROXY='http://y00951466:Yzy%401454662387@<反代ip>:6688'

export http_proxy="$PROXY"
export https_proxy="$PROXY"
export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"

export NO_CHECK_CERTIFICATE=true # https不检查证书  
export no_proxy=127.0.0.1,*.huawei.com,localhost,local,.local,inhuawei.com  
export GIT_SSL_NO_VERIFY=true

git config --global http.proxy "$PROXY"
git config --global https.proxy "$PROXY"
git config --global user.email "zhenyang0319@gmail.com"
git config --global user.name "pachi33"