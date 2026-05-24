for d in data/posco/train/*; do
    if [ -d "$d" ]; then
        echo "$d: $(find "$d" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)"
    fi
done
