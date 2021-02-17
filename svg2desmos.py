import pyperclip
from xml.dom import minidom
from math import cos, sin, sqrt, atan2, pi
import numpy as np
import re



def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb
    

def py_ang(v1, v2): #finds angle between 2 vectors
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def sign(x):
    if x == 0:
        return 0
    else:
        return abs(x)/x

def negative_zero_clamp(x):
    if -0.01<x<0:
        return 0 
    else:
        return x
    
def angle_two_vecs(u, v):
    sign_xxx = sign(u[0][0] * v[1][0] - u[1][0] * v[0][0])
    u = np.transpose(u)[0]
    v = np.transpose(v)[0]
    h = (u.dot(v))/ ( np.linalg.norm(u) * np.linalg.norm(v))
    if h > 1:   
        h = 1
    elif h < -1:
        h = -1
    return sign_xxx * np.arccos(h)

cpx = 0
cpy = 0
spx = 0
spy = 0
f = 1
default_colour = '#000000'
only_fill = True  
only_lines = False


class path:
    def parse_path(self, path):
        global cpx
        global cpy
        global spx
        global spy
        global f
        self.path = path
        paths_list = []
        letters_list = []
        not_fill = []
        fill = []
        not_stroke = []
        stroke = []
        doc = minidom.parse(self.path)
        path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
        colour_strings = [[i,path.getAttribute('fill')] for i, path in enumerate(doc.getElementsByTagName('path'))]
        stroke_strings = [[i,path.getAttribute('stroke')] for i, path in enumerate(doc.getElementsByTagName('path'))]

        for i in range(len(stroke_strings)):
            if stroke_strings[i][1] == '':
                not_stroke.append(i)
            else:
                stroke.append(i)

        for i in range(len(colour_strings)):
            if colour_strings[i][1] == '':
                not_fill.append(i)
            else:
                fill.append(i)

        for i in range(len(colour_strings)):
            if colour_strings[i][1] == '':
                if stroke_strings[i][1] != '':
                    colour_strings[i][1] = stroke_strings[i][1]
                else:
                    colour_strings[i][1] = default_colour
            if stroke_strings[i][1] == '':
                if colour_strings[i][1] != '':
                    stroke_strings[i][1] = colour_strings[i][1]
                else:
                    stroke_strings[i][1] = default_colour      

        TOKEN_RE = re.compile("[MmZzLlHhVvCcSsQqTtAa]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
        def _tokenize_path_replace(pathdef):
            return TOKEN_RE.findall(pathdef)
        amount_of_paths = len(path_strings)
        for lista in path_strings:
            paths_list.append(_tokenize_path_replace(lista))
        command = [[] for i in range(amount_of_paths)]
        for i in range(len(paths_list)):
            j=0
            loop_count = len(paths_list[i])
            while j < loop_count:
                if paths_list[i][j] == 'C':
                    letters_list.append('C')
                    command[i].append(self.cubic(cpx,-f*float(cpy),paths_list[i][j+1],-f*float(paths_list[i][j+2]),paths_list[i][j+3],-f*float(paths_list[i][j+4]),paths_list[i][j+5],-f*float(paths_list[i][j+6]), colour_strings[i][1]))
                    cpx = paths_list[i][j+5]
                    cpy = paths_list[i][j+6]
                    try:
                        if not str(paths_list[i][j+7]).isalpha():
                            paths_list[i].insert(j+7,'C')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'Q':
                    letters_list.append('Q')
                    command[i].append(self.quadratic(cpx,-f*float(cpy),paths_list[i][j+1],-f*float(paths_list[i][j+2]),paths_list[i][j+3],-f*float(paths_list[i][j+4]),colour_strings[i][1]))
                    cpx = paths_list[i][j+3]
                    cpy = paths_list[i][j+4]
                    try:
                        if not str(paths_list[i][j+5]).isalpha():
                            paths_list[i].insert(j+5,'Q')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'M':
                    letters_list.append('M')
                    cpx = paths_list[i][j+1]
                    cpy = float(paths_list[i][j+2])
                    if j == 0:
                        spx = cpx
                        spy = cpy
                    try:
                        if not str(paths_list[i][j+3]).isalpha():
                            paths_list[i].insert(j+3,'L')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'L':
                    letters_list.append('L')
                    command[i].append(self.line(cpx,-f*float(cpy),paths_list[i][j+1],-f*float(paths_list[i][j+2]),colour_strings[i][1]))
                    cpx = paths_list[i][j+1]
                    cpy = paths_list[i][j+2]
                    try:
                        if not str(paths_list[i][j+3]).isalpha():
                            paths_list[i].insert(j+3,'L')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'Z' :
                    letters_list.append('Z')
                    command[i].append(self.line(cpx,-f*float(cpy),spx,-f*float(spy),colour_strings[i][1]))
                    cpx = spx
                    cpy = spy
                    spx = float(spx)
                    spy = float(spy)
                    if j < len(paths_list[i]) - 1:
                        if paths_list[i][j+1] == 'M':
                            spx = paths_list[i][j+2]
                            spy = paths_list[i][j+3]
                        elif paths_list[i][j+1] == 'm':
                            spx += float(paths_list[i][j+2])
                            spy += float(paths_list[i][j+3])

                elif paths_list[i][j] == 'H':
                    letters_list.append('H')
                    command[i].append(self.line(cpx,-f*float(cpy),paths_list[i][j+1],-f*float(cpy),colour_strings[i][1]))
                    cpx = paths_list[i][j+1]
                    try:
                        if not str(paths_list[i][j+2]).isalpha():
                            paths_list[i].insert(j+2,'H')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'V':
                    letters_list.append('V')
                    command[i].append(self.line(cpx,-f*float(cpy),cpx, -f*float(paths_list[i][j+1]), colour_strings[i][1]))
                    cpy = paths_list[i][j+1]
                    try:
                        if not str(paths_list[i][j+2]).isalpha():
                            paths_list[i].insert(j+2,'V')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'S':
                    if letters_list[-1] == 'C' or letters_list[-1] == 'S' :
                        command[i].append(self.cubic(cpx,-f*float(cpy),paths_list[i][j-4],-f*float(paths_list[i][j-3]),paths_list[i][j+1],-f*float(paths_list[i][j+2]),paths_list[i][j+3],-f*float(paths_list[i][j+4]),colour_strings[i][1]))
                    else:
                        command[i].append(self.cubic(cpx,-f*float(cpy),cpx,-f*float(cpy),paths_list[i][j+1],-f*float(paths_list[i][j+2]),paths_list[i][j+3], -f*float(paths_list[i][j+4]), colour_strings[i][1]))
                    cpx = paths_list[i][j+3]
                    cpy = paths_list[i][j+4]
                    try:
                        if not str(paths_list[i][j+5]).isalpha():
                            paths_list[i].insert(j+5,'S')
                            loop_count+=1
                    except:
                        IndexError
                    letters_list.append('S')   
                elif paths_list[i][j] == 'T':
                    if letters_list[-1] == 'Q' or letters_list[-1] == 'T':
                        command[i].append(self.quadratic(cpx,-f*float(cpy),paths_list[i][j-4],-f*float(paths_list[i][j-3]),paths_list[i][j+1],-f*float(paths_list[i][j+2]), colour_strings[i][1]))
                    else:
                        command[i].append(self.quadratic(cpx,-f*float(cpy),cpx,-f*float(cpy),paths_list[i][j+1],-f*float(paths_list[i][j+2]),colour_strings[i][1]))
                    cpx = paths_list[i][j+1]
                    cpy = paths_list[i][j+2]
                    try:
                        if not str(paths_list[i][j+3]).isalpha():
                            paths_list[i].insert(j+3,'T')
                            loop_count+=1
                    except:
                        IndexError
                    letters_list.append('T')
                elif paths_list[i][j] == 'A':
                    letters_list.append('A')
                    command[i].append(self.arc(cpx,-f*float(cpy),float(paths_list[i][j+1]),float(paths_list[i][j+2]),float(paths_list[i][j+3]),float(paths_list[i][j+4]),float(paths_list[i][j+5]),float(paths_list[i][j+6]), -f*float(paths_list[i][j+7]),colour_strings[i][1]))
                    cpx = paths_list[i][j+6]
                    cpy = paths_list[i][j+7]
                    try:
                        if not str(paths_list[i][j+8]).isalpha():
                            paths_list[i].insert(j+8,'A')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'c':
                    letters_list.append('c')
                    cpx = float(cpx)
                    cpy = float(cpy)
                    command[i].append(self.cubic(cpx,-f*float(cpy),cpx + float(paths_list[i][j+1]),-f*(cpy + float(paths_list[i][j+2])),cpx + float(paths_list[i][j+3]),-f*(cpy + float(paths_list[i][j+4])), cpx+ float(paths_list[i][j+5]),-f*(cpy + float(paths_list[i][j+6])), colour_strings[i][1]))
                    cpx += float(paths_list[i][j+5])
                    cpy += float(paths_list[i][j+6])
                    try:
                        a = str((paths_list[i][j+7])).isalpha()
                        if not a:
                            paths_list[i].insert(j+7,'c')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'q':
                    letters_list.append('q')
                    cpx = float(cpx)
                    cpy = float(cpy)
                    command[i].append(self.quadratic(cpx,-f*float(cpy),cpx + float(paths_list[i][j+1]),-f*(cpy+float(paths_list[i][j+2])),cpx + float(paths_list[i][j+3]),-f*(cpy+float(paths_list[i][j+4])),colour_strings[i][1]))
                    cpx += float(paths_list[i][j+3])
                    cpy += float(paths_list[i][j+4])
                    try:
                        if not str(paths_list[i][j+5]).isalpha():
                            paths_list[i].insert(j+5,'q')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'm':
                    cpx = float(cpx)
                    cpy = float(cpy)
                    spx = float(spx)
                    spy = float(spy)
                    letters_list.append('m')
                    if j != 0:
                        cpx += float(paths_list[i][j+1])
                        cpy += float(paths_list[i][j+2])
                    if j == 0:
                        spx = float(paths_list[i][j+1])
                        spy = float(paths_list[i][j+2])
                        cpx = float(paths_list[i][j+1])
                        cpy = float(paths_list[i][j+2])
                    try:
                        if not str(paths_list[i][j+3]).isalpha():
                            paths_list[i].insert(j+3,'l')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'l':
                    cpx = float(cpx)
                    cpy = float(cpy)
                    letters_list.append('l')
                    command[i].append(self.line(cpx,-f*float(cpy),cpx + float(paths_list[i][j+1]),-f*(cpy+float(paths_list[i][j+2])),colour_strings[i][1]))
                    cpx += float(paths_list[i][j+1])
                    cpy += float(paths_list[i][j+2])
                    try:
                        if not str(paths_list[i][j+3]).isalpha():
                            paths_list[i].insert(j+3,'l')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'h':
                    cpx = float(cpx)
                    letters_list.append('h')
                    command[i].append(self.line(cpx,-f*float(cpy),cpx+float(paths_list[i][j+1]),-f*float(cpy),colour_strings[i][1]))
                    cpx += float(paths_list[i][j+1])
                    try:
                        if not str(paths_list[i][j+2]).isalpha():
                            paths_list[i].insert(j+2,'h')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'v':
                    cpy = float(cpy)
                    letters_list.append('v')
                    command[i].append(self.line(cpx,-f*float(cpy), cpx, -(cpy + float(paths_list[i][j+1])), colour_strings[i][1]))
                    cpy += float(paths_list[i][j+1])
                    try:
                        if not str(paths_list[i][j+2]).isalpha():
                            paths_list[i].insert(j+2,'v')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'a':
                    cpx = float(cpx)
                    cpy = float(cpy)
                    letters_list.append('a')
                    command[i].append(self.arc(cpx,-f*cpy,float(paths_list[i][j+1]),float(paths_list[i][j+2]),float(paths_list[i][j+3]),float(paths_list[i][j+4]),float(paths_list[i][j+5]),cpx+float(paths_list[i][j+6]),-f*(cpy+float(paths_list[i][j+7])),colour_strings[i][1]))
                    cpx += float(paths_list[i][j+6])
                    cpy += float(paths_list[i][j+7])
                    try:
                        if not str(paths_list[i][j+8]).isalpha():
                            paths_list[i].insert(j+8,'a')
                            loop_count+=1
                    except:
                        IndexError

                elif paths_list[i][j] == 's':
                    cpx = float(cpx)
                    cpy = float(cpy)
                    
                    if letters_list[-1] == 's' or letters_list[-1] == 'c':
                        ppx = cpx - float(paths_list[i][j-2]) 
                        ppy = cpy - float(paths_list[i][j-1]) 
                        command[i].append(self.cubic(cpx,-f*float(cpy),2*cpx-(float(paths_list[i][j-4])+ppx),-f*2*cpy+(float(paths_list[i][j-3])+ppy),cpx+float(paths_list[i][j+1]),-f*(cpy+float(paths_list[i][j+2])),cpx+float(paths_list[i][j+3]),-f*(cpy+float(paths_list[i][j+4])),colour_strings[i][1]))
                    else:
                        command[i].append(self.cubic(cpx,-f*float(cpy),cpx,-f*float(cpy),cpx+float(paths_list[i][j+1]),-f*(cpy+float(paths_list[i][j+2])),cpx+float(paths_list[i][j+3]), -f*(cpy+float(paths_list[i][j+4])), colour_strings[i][1]))
                    cpx += float(paths_list[i][j+3])
                    cpy += float(paths_list[i][j+4])
                    try:
                        if not str(paths_list[i][j+5]).isalpha():
                            paths_list[i].insert(j+5,'s')
                            loop_count+=1
                    except:
                        IndexError
                    letters_list.append('s')
                elif paths_list[i][j] == 'z' :
                    spx = float(spx)
                    spy = float(spy)
                    letters_list.append('z')
                    command[i].append(self.line(cpx,-f*float(cpy),spx,-f*float(spy),colour_strings[i][1]))
                    cpx = spx
                    cpy = spy
                    if j < len(paths_list[i]) - 1:
                        if paths_list[i][j+1] == 'm':
                            spx += float(paths_list[i][j+2])
                            spy += float(paths_list[i][j+3])
                        elif paths_list[i][j+1] == 'M':
                            spx = paths_list[i][j+2]
                            spy = paths_list[i][j+3]
                
                j+=1    

        for i in range(len(letters_list)-1,-1,-1):
            if letters_list[i].lower() == 'm':
                del letters_list[i]

        
        for i in range(len(letters_list)):
            letters_list[i] = f'{i+1}:{letters_list[i]}'
        
        command_string = ''
        
        #print(letters_list)    
        #print(colour_strings)            
        print("Number Of Functions: " + str(len(letters_list)))

        
        #for command_path in command:
        #    command_string += "Calc.setExpression({latex: \'" + command_path[0] + ',' + command_path[1] + '\', color: \'#000000\'})'+  '\n'
        #print(colour_strings)
        if not only_lines:
            for j in range(0,len(command),1):
                inner_x = ''
                inner_y = ''
                for i in range(len(command[j])):
                    i_x = str(command[j][i][0]).replace("sqrt","sqr").replace("left","lef").replace("right","righ").replace("arctan","aan").replace('t',f"(t-{i})").replace("lef","left").replace("righ","right").replace("aan","arctan").replace("sqr","sqrt")
                    i_y = str(command[j][i][1]).replace("sqrt","sqr").replace("left","lef").replace("right","righ").replace("arctan","aan").replace('t',f"(t-{i})").replace("lef","left").replace("righ","right").replace("aan","arctan").replace("sqr","sqrt")

                    if i != len(command[j]) - 1:
                        inner_x += (f"{i}<t<{i+1}:" + i_x + ",")
                        inner_y += (f"{i}<t<{i+1}:" + i_y + ",")
                    else:
                        inner_x += (f"{i}<t<{i+1}:" + i_x + "")
                        inner_y += (f"{i}<t<{i+1}:" + i_y + "")
                upper_bound = 18 * ((len(command[j])*2)//18 + 1)
                if j in not_fill:
                    fill_true_false = 'false'
                else:
                    fill_true_false = 'true'
                
                command_string += "Calc.setExpression({latex: \'" + "\\\left(\\\left\\\{" + inner_x + "\\\\right\\\},\\\left\\\{" + inner_y + "\\\\right\\\}\\\\right)" +  '\',parametricDomain : { min : \'0\', max : \'' + str(upper_bound)+ '\'}' +  f',fill : {fill_true_false}, fillOpacity : 1.0 , color: \'' +colour_strings[j][1] + '\'})' + '\n'
        

        if not only_fill:
            for j in range(len(command)):
                for i in range(len(command[j])):
                    
                    command_string += "Calc.setExpression({latex: \'" + "(" + command[j][i][0] + "," + command[j][i][1] + ")" +  '\', color: \'' + stroke_strings[j][1] + '\'})' + '\n'
        pyperclip.copy(command_string)
        
    def cubic(self, sx, sy, csx, csy, cex, cey, ex, ey, colour):
        self.sx = ex
        self.sy = ey
        self.csx = cex
        self.csy = cey
        self.cex = csx
        self.cey = csy
        self.ex = sx
        self.ey = sy
        self.colour = colour
        x_command = f"\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.ex})+t({self.cex})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.cex})+t({self.csx})\\\\right)\\\\right)+t\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.cex})+t({self.csx})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.csx})+t({self.sx})\\\\right)\\\\right)"
        y_command = f"\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.ey})+t({self.cey})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.cey})+t({self.csy})\\\\right)\\\\right)+t\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.cey})+t({self.csy})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.csy})+t({self.sy})\\\\right)\\\\right)"
        return x_command,y_command

    def quadratic(self, sx, sy, cx, cy, ex, ey, colour):
        self.sx = ex
        self.sy = ey
        self.cx = cx
        self.cy = cy
        self.ex = sx
        self.ey = sy
        self.colour = colour
        x_command = f"\\\left(1-t\\\\right)^{{2}}({self.sx})+2\\\left(1-t\\\\right)t({self.cx})+\\\left(t\\\\right)^{{2}}({self.ex})"
        y_command = f"\\\left(1-t\\\\right)^{{2}}({self.sy})+2\\\left(1-t\\\\right)t({self.cy})+\\\left(t\\\\right)^{{2}}({self.ey})"
        return x_command,y_command

    def move(self, x, y, rel):
        if rel == 0:
            cpx = x
            cpy = y
        else:
            cpx += x
            cpy += y

    def line(self, sx, sy, ex, ey, colour):
        self.sx = ex
        self.sy = ey
        self.ex = sx
        self.ey = sy
        self.colour = colour
        #\left(,\right)
        x_command = f"({self.ex})\\\left(1-t\\\\right)+({self.sx})t"
        y_command = f"({self.ey})\\\left(1-t\\\\right)+({self.sy})t"
        return x_command,y_command
        
    def arc(self, sx, sy, rx, ry, angle, large_arc, sweep, ex, ey, colour):
        self.sx = ex 
        self.sy = ey
        self.rx = rx 
        self.ry = ry
        self.angle = angle 
        self.large_arc = large_arc
        self.sweep = not sweep 
        self.ex = sx 
        self.ey = sy
        self.colour = colour
        xy_subtracty_thing = np.array([[(sx-ex)/2], [(sy - ey) / 2]]) 
        xy1_prime_rotation_mtx = np.array([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]]) 
        xy1_prime = xy1_prime_rotation_mtx.dot(xy_subtracty_thing)
        x1_p = xy1_prime[0][0]
        y1_p = xy1_prime[1][0]
        bigass_sqrt_positive_for_cxy1 = sqrt( negative_zero_clamp((rx**2 * ry**2 - rx**2 * y1_p**2 - ry**2 * x1_p**2)/ (rx**2 * y1_p**2 + ry**2 * x1_p**2)) )
        signh = -1 if self.large_arc == self.sweep else 1 
        bigass_sqrt_for_cxy1 = signh * bigass_sqrt_positive_for_cxy1 
        vector_thing_for_cxy_prime = np.array([[ (rx * y1_p)/ry], [(-ry * x1_p) / rx] ]) 
        cxy_prime = bigass_sqrt_for_cxy1 * vector_thing_for_cxy_prime 
        cxy_rotation_mtx = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        xy12_midpoints = np.array([[(sx + ex)/2], [(sy + ey)/2]])
        cxy = cxy_rotation_mtx.dot(cxy_prime) + xy12_midpoints 
        cx_p = cxy_prime[0][0]
        cy_p = cxy_prime[1][0]
        xy1p_minus_cxyp_div_rxy = np.array([[(x1_p - cx_p)/rx], [(y1_p - cy_p)/ry]])
        theta1 = angle_two_vecs( np.array([[1], [0]]), xy1p_minus_cxyp_div_rxy )
        minus_xy1p_minus_cxyp_div_rxy = np.array([[(-x1_p - cx_p)/rx], [(-y1_p - cy_p)/ry]])
        delta_theta = angle_two_vecs(xy1p_minus_cxyp_div_rxy, minus_xy1p_minus_cxyp_div_rxy ) % (2 * pi)
        if self.sweep == 0 and delta_theta > 0:
            delta_theta -= 2*pi
        elif self.sweep == 1 and delta_theta < 0:
            delta_theta += 2*pi 
        theta_2 = theta1 + delta_theta
        theta_max = max(theta1,theta_2)
        theta_min = min(theta1,theta_2)
        c_x = cxy[0][0]
        c_y = cxy[1][0]
        max_min = theta_max - theta_min
        if max_min == 0:
            max_min = pi
       
        command_x = f"(-{self.ry})\\\sin\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\cos\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({c_x})"
        command_y = f"({self.ry})\\\cos\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\sin\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({c_y})"
        return command_x,command_y
        

        



        
c = path()
c.parse_path('desmos\svgs\cards.svg')
