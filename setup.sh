mkdir -p data/valid
gdown https://drive.google.com/u/0/uc?id=1_d1i1NDso6aFJ-FnuGZasMoGA7RYDxNx
unzip valid.zip >> /dev/null
mv audio_1.wav data/valid
mv audio_2.wav data/valid
mv audio_3.wav data/valid
rm valid.zip