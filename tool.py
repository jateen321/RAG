import sys

'''
python tool.py greet "Jateen"     → "Hello, Jateen!"
python tool.py reverse "hello"    → "olleh"
python tool.py count "file.txt"   → "42 words"
'''
command = sys.argv[1]

if command ==  "count":
    filename = sys.argv[2]
    with open(filename, "r") as f:
        text = f.read()
        words = text.split()
        print(f"{len(words)} words")

elif command == "reverse":
    text = sys.argv[2]
    print(text[::-1])

elif command == "greet":
    name = sys.argv[2]
    print(f"Hello, {name}!")

else:
    print("Unknown command")