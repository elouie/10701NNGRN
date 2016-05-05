f = open("model_BNG_out_3.txt", "r")
o = open("model_BNG_out_4.txt", "w")

newl = ""
for line in f:
   if line[0] == "R":
     o.write(line)
   elif line[0] == '[':
     newl = line[1:].strip() + " "
   else:
     newl = newl + line.strip()
     if newl[-1] == ']':
       newl = newl[:-1]
       o.write(newl + "\n")
     else:
       newl = newl + " "
