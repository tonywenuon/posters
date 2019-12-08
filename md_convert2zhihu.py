import re

src_file_name = 'bert2_xlnet'
src_file = src_file_name + '.md'
tar_file = src_file_name + '_2.md'

def repl1(m):
    inner_word = m.group(0) 
    return '<br><br>' + inner_word.strip() + '<br><br>'


def repl2(m):
    inner_word = m.group(1) 
    #return '<br><br>$' + inner_word.strip() + '$<br><br>'
    remove_space = ''
    for ch in inner_word:
        if ch != ' ':
            remove_space += ch
    return '<br><br>![](https://latex.codecogs.com/gif.latex?' + remove_space.strip() + ')<br><br>'

def repl3(m):
    inner_word = m.group(1) 
    #return '<br><br>$' + inner_word.strip() + '$<br><br>'
    remove_space = ''
    for ch in inner_word:
        if ch != ' ':
            remove_space += ch
    return '![](https://latex.codecogs.com/gif.latex?' + remove_space.strip() + ')'

with open(src_file, 'r') as f_read:
    text = f_read.readlines()
    for k, item in enumerate(text):
        after_rep = re.sub(r'\!\[(.*?)\)', repl1, item)
        after_rep = re.sub(r'\$\$(.*?)\$\$', repl2, after_rep)
        after_rep = re.sub(r'\$(.*?)\$', repl3, after_rep)
        text[k] = after_rep
    
    with open(tar_file, 'w') as f_write:
        f_write.writelines(text)
