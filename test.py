Invoke-WebRequest `
  -Uri "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz" `
  -OutFile "ffmpeg-release-amd64-static.tar.xz"

echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
