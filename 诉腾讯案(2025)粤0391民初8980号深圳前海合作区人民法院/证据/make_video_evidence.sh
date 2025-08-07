ffmpeg -r 0.1 -loop 1  -y -f image2 -s 2712x1220  -i video_images/%02d.jpg  -i video_images/a.mp3 -vframes 300   -vcodec libx264  -crf 25 -pix_fmt yuv420p -acodec copy  tencent.mp4
