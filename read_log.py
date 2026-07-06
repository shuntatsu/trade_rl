try:
    with open("evolution_log.txt", "rb") as f:
        content_bytes = f.read()

    try:
        content = content_bytes.decode("utf-8")
    except:
        content = content_bytes.decode("cp932", errors="ignore")

    print(content)

except Exception as e:
    print(f"Error reading log: {e}")
