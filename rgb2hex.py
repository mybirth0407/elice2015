def main ():
    r = input() # First line
    g = input() # Second line
    b = input() # Third line

    # change r, g, b into integer...
    
    print(rgb2hex(r, g, b))

    
def rgb2hex(r, g, b):
    hex_color = "#%02X%02X%02X" % (int(r), int(g), int(b))
    
    return hex_color

if __name__ == "__main__":
    main()