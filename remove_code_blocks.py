import sys

# loop through args (the first arg is the name of this file)
for arg in sys.argv[1:]:
    # try-except catches if file doesn't exist
    try:
        # get lines
        with open(arg, 'r') as f:
            lines = f.readlines()
        
        # filter out lines that begin with comments
        no_comments = list(filter(lambda s: s[0] != '#', lines))

        # write lines back to file
        with open(arg, 'w+') as g:
            for line in no_comments:
                g.write(line)

    # handle exeption
    except IOError:
        print(f"File {arg} not accessible")
