import re
def repl(m):
    inner_word = m.group(1) 
    #return '<br><br>$' + inner_word.strip() + '$<br><br>'
    remove_space = ''
    for ch in inner_word:
        if ch != ' ':
            remove_space += ch
    return '<br><br>![](https://latex.codecogs.com/gif.latex?' + remove_space.strip() + ')</br></br>'

with open('Important1_seq2seq_attention.md', 'r') as f_read:
    text = f_read.readlines()
    for k, item in enumerate(text):
        after_rep = re.sub(r'\$\$(.*?)\$\$', repl, item)
        after_rep = re.sub(r'\$(.*?)\$', repl, after_rep)
        text[k] = after_rep
    
    with open('2.md', 'w') as f_write:
        f_write.writelines(text)
