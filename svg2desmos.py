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


class path:
    def parse_path(self, path):
        global cpx
        global cpy
        global spx
        global spy
        self.path = path
        paths_list = []
        letters_list = []
        doc = minidom.parse(self.path)
        path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
        TOKEN_RE = re.compile("[MmZzLlHhVvCcSsQqTtAa]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
        def _tokenize_path_replace(pathdef):
            return TOKEN_RE.findall(pathdef)
        amount_of_paths = len(path_strings)
        for lista in path_strings:
            paths_list.append(_tokenize_path_replace(lista))
        command = ''
        for i in range(len(paths_list)):
            j=0
            loop_count = len(paths_list[i])
            while j < loop_count:
                if paths_list[i][j] == 'C':
                    letters_list.append('C')
                    command += self.cubic(cpx,-float(cpy),paths_list[i][j+1],-float(paths_list[i][j+2]),paths_list[i][j+3],-float(paths_list[i][j+4]),paths_list[i][j+5],-float(paths_list[i][j+6]), 1) + '\n'
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
                    command += self.quadratic(cpx,-float(cpy),paths_list[i][j+1],-float(paths_list[i][j+2]),paths_list[i][j+3],-float(paths_list[i][j+4]),1 ) + '\n'
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
                        spx = paths_list[i][j+1]
                        spy = paths_list[i][j+2]
                    try:
                        if not str(paths_list[i][j+3]).isalpha():
                            paths_list[i].insert(j+3,'L')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'L':
                    letters_list.append('L')
                    command += self.line(cpx,-float(cpy),paths_list[i][j+1],-float(paths_list[i][j+2]),1) + '\n'
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
                    command += self.line(cpx,-float(cpy),spx,-float(spy),0) + '\n'
                    cpx = spx
                    cpy = spy
                    if j < len(paths_list[i]) - 1:
                        if paths_list[i][j+1] == 'M':
                            spx = paths_list[i][j+2]
                            spy = paths_list[i][j+3]
                elif paths_list[i][j] == 'H':
                    letters_list.append('H')
                    command += self.horizontal(cpx, -float(cpy), paths_list[i][j+1], 1) + '\n'
                    cpx = paths_list[i][j+1]
                    try:
                        if not str(paths_list[i][j+2]).isalpha():
                            paths_list[i].insert(j+2,'H')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'V':
                    letters_list.append('V')
                    command += self.vertical(cpx, -float(cpy),-float(paths_list[i][j+1]), 1 ) + '\n'
                    cpy = paths_list[i][j+1]
                    try:
                        if not str(paths_list[i][j+2]).isalpha():
                            paths_list[i].insert(j+2,'V')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'S':
                    if letters_list[-1] == 'C' or letters_list[-1] == 'S' :
                        command += self.cubic(cpx,-float(cpy),paths_list[i][j-4],-float(paths_list[i][j-3]),paths_list[i][j+1],-float(paths_list[i][j+2]),paths_list[i][j+3],-float(paths_list[i][j+4]),1) + '\n'
                    else:
                        command += self.cubic(cpx,-float(cpy),cpx,-float(cpy),paths_list[i][j+1],-float(paths_list[i][j+2]),paths_list[i][j+3], -float(paths_list[i][j+4]), 1 ) + '\n'
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
                        command += self.quadratic(cpx,-float(cpy),paths_list[i][j-4],-float(paths_list[i][j-3]),paths_list[i][j+1],paths_list[i][j+2], 0) + '\n'
                    else:
                        command += self.quadratic(cpx,-float(cpy),cpx,-float(cpy),paths_list[i][j+1],-float(paths_list[i][j+2]),1) + '\n'
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
                    command += self.arc(cpx,-float(cpy),float(paths_list[i][j+1]),float(paths_list[i][j+2]),float(paths_list[i][j+3]),float(paths_list[i][j+4]),float(paths_list[i][j+5]),float(paths_list[i][j+6]), -float(paths_list[i][j+7])) + '\n'
                    cpx = paths_list[i][j+6]
                    cpy = paths_list[i][j+7]
                elif paths_list[i][j] == 'c':
                    letters_list.append('c')
                    cpx = float(cpx)
                    cpy = float(cpy)
                    command += self.cubic(cpx,-float(cpy),cpx + float(paths_list[i][j+1]),-(cpy + float(paths_list[i][j+2])),cpx + float(paths_list[i][j+3]),-(cpy + float(paths_list[i][j+4])), cpx+ float(paths_list[i][j+5]),-(cpy + float(paths_list[i][j+6])), 1) + '\n'
                    cpx += float(paths_list[i][j+5])
                    cpy += float(paths_list[i][j+6])
                    a = str((paths_list[i][j+7])).isalpha()
                    try:
                        if not a:
                            paths_list[i].insert(j+7,'c')
                            loop_count+=1
                    except:
                        IndexError
                elif paths_list[i][j] == 'q':
                    letters_list.append('q')
                    cpx = float(cpx)
                    cpy = float(cpy)
                    command += self.quadratic(cpx,-float(cpy),cpx + float(paths_list[i][j+1]),-(cpy+float(paths_list[i][j+2])),cpx + float(paths_list[i][j+3]),-(cpy+float(paths_list[i][j+4])),1 ) + '\n'
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
                    command += self.line(cpx,-float(cpy),cpx + float(paths_list[i][j+1]),-(cpy+float(paths_list[i][j+2])),1) + '\n'
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
                    command += self.horizontal(cpx, -float(cpy), cpx + float(paths_list[i][j+1]), 1) + '\n'
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
                    command += self.vertical(cpx, -float(cpy),-(cpy + float(paths_list[i][j+1])), 1 ) + '\n'
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
                    command += self.arc(cpx,-cpy,float(paths_list[i][j+1]),float(paths_list[i][j+2]),float(paths_list[i][j+3]),float(paths_list[i][j+4]),float(paths_list[i][j+5]),cpx+float(paths_list[i][j+6]),-(cpy+float(paths_list[i][j+7]))) + '\n'
                    cpx += float(paths_list[i][j+6])
                    cpy += float(paths_list[i][j+7])
                elif paths_list[i][j] == 's':
                    cpx = float(cpx)
                    cpy = float(cpy)
                    
                    if letters_list[-1] == 's' or letters_list[-1] == 'c':
                        ppx = cpx - float(paths_list[i][j-2]) 
                        ppy = cpy - float(paths_list[i][j-1]) 
                        command += self.cubic(cpx,-float(cpy),2*cpx-(float(paths_list[i][j-4])+ppx),-2*cpy+(float(paths_list[i][j-3])+ppy),cpx+float(paths_list[i][j+1]),-(cpy+float(paths_list[i][j+2])),cpx+float(paths_list[i][j+3]),-(cpy+float(paths_list[i][j+4])),1) + '\n'
                    else:
                        command += self.cubic(cpx,-float(cpy),cpx,-float(cpy),cpx+float(paths_list[i][j+1]),-(cpy+float(paths_list[i][j+2])),cpx+float(paths_list[i][j+3]), -(cpy+float(paths_list[i][j+4])), 1 ) + '\n'
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
                    command += self.line(cpx,-float(cpy),spx,-float(spy),0) + '\n'
                    cpx = spx
                    cpy = spy
                    if j < len(paths_list[i]) - 1:
                        if paths_list[i][j+1] == 'm':
                            spx += float(paths_list[i][j+2])
                            spy += float(paths_list[i][j+3])
                
                j+=1    
                
        print(letters_list)
        pyperclip.copy(command)

        
    def cubic(self, sx, sy, csx, csy, cex, cey, ex, ey, colour):
        self.sx = sx
        self.sy = sy
        self.csx = csx
        self.csy = csy
        self.cex = cex
        self.cey = cey
        self.ex = ex
        self.ey = ey
        self.colour = colour
        command = f"Calc.setExpression({{latex: '\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.ex})+t({self.cex})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.cex})+t({self.csx})\\\\right)\\\\right)+t\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.cex})+t({self.csx})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.csx})+t({self.sx})\\\\right)\\\\right),\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.ey})+t({self.cey})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.cey})+t({self.csy})\\\\right)\\\\right)+t\\\left(\\\left(1-t\\\\right)\\\left(\\\left(1-t\\\\right)({self.cey})+t({self.csy})\\\\right)+t\\\left(\\\left(1-t\\\\right)({self.csy})+t({self.sy})\\\\right)\\\\right)\\\\right)', color: '{'#000000'}'}})"
        return command

    def quadratic(self, sx, sy, cx, cy, ex, ey, colour):
        self.sx = sx
        self.sy = sy
        self.cx = cx
        self.cy = cy
        self.ex = ex
        self.ey = ey
        self.colour = colour
        command = f"Calc.setExpression({{latex: '\\\left(\\\left(1-t\\\\right)^{{2}}({self.sx})+2\\\left(1-t\\\\right)t({self.cx})+\\\left(t\\\\right)^{{2}}({self.ex}),\\\left(1-t\\\\right)^{{2}}({self.sy})+2\\\left(1-t\\\\right)t({self.cy})+\\\left(t\\\\right)^{{2}}({self.ey})\\\\right)', color: '{'#000000'}'}})"
        return command

    def move(self, x, y, rel):
        if rel == 0:
            cpx = x
            cpy = y
        else:
            cpx += x
            cpy += y

    def line(self, sx, sy, ex, ey, colour):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.colour = colour
        command = f"Calc.setExpression({{latex: '\\\left(\\\sqrt{{\\\left({self.ey}-{self.sy}\\\\right)^{{2}}+\\\left({self.ex}-{self.sx}\\\\right)^{{2}}}}\\\cos\\\left(\\\\arctan\\\left({self.ey}-{self.sy},{self.ex}-{self.sx}\\\\right)\\\\right)t+{self.sx},\\\sqrt{{\\\left({self.ey}-{self.sy}\\\\right)^{{2}}+\\\left({self.ex}-{self.sx}\\\\right)^{{2}}}}\\\sin\\\left(\\\\arctan\\\left({self.ey}-{self.sy},{self.ex}-{self.sx}\\\\right)\\\\right)t+{self.sy}\\\\right)', color: '{'#000000'}'}})"
        return command

    def horizontal(self, sx, sy, ex, colour):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.colour = colour
        x_delta = float(self.ex) - float(self.sx)
        command = f"Calc.setExpression({{latex: '\\\left(\\\left({x_delta}\\\\right)t+{self.sx},+{self.sy}\\\\right)', color: '{'#000000'}'}})"
        return command

    def vertical(self, sx, sy, ey, colour):
        self.sx = sx
        self.sy = sy
        self.ey = ey
        self.colour = colour
        y_delta = float(self.ey) - float(self.sy)
        command = f"Calc.setExpression({{latex: '\\\left({self.sx},\\\left({y_delta}\\\\right)t+{self.sy}\\\\right)', color: '{'#000000'}'}})"
        return command
        
        '''def arc(self, sx, sy, rx, ry, angle, large_arc, sweep, ex, ey):
        self.sx = sx
        self.sy = sy
        self.rx = rx
        self.ry = ry
        self.angle = angle
        self.large_arc = large_arc
        self.sweep = sweep
        self.ex = ex
        self.ey = ey
        if self.sweep == 1:
            self.sweep = 0 
        else:
            self.sweep = 1  
        x1_prime = (self.sx-self.ex)/2 * cos(self.angle) + (self.sy-self.ey)/2 * sin(self.angle)
        y1_prime = (self.sy-self.ey)/2 * cos(self.angle) - (self.sx-self.ex)/2 * sin(self.angle)
        signh = -1 if self.large_arc == self.sweep else 1
        C = signh * sqrt(negative_zero_clamp((self.rx**2 * self.ry**2 - self.rx**2 * y1_prime**2 - self.ry**2 * x1_prime**2)/ (self.rx**2 * y1_prime**2 + self.ry**2 * x1_prime**2)))
        cx_prime = C * self.rx * y1_prime / self.ry
        cy_prime = -C * self.ry * x1_prime / self.rx
        c_x = cx_prime * cos(self.angle) - cy_prime * sin(self.angle) + (self.sx + self.ex)/2
        c_y = cx_prime * sin(self.angle) + cy_prime * cos(self.angle) + (self.sy + self.ey)/2
        s = sign((y1_prime-cy_prime)/self.ry)
        theta_1 = s * py_ang([(x1_prime-cx_prime)/self.rx, (y1_prime-cy_prime)/self.ry],[1,0])
        ux = (x1_prime - cx_prime) / self.rx
        uy = (y1_prime - cy_prime) / self.ry    
        vx = (-x1_prime-cx_prime) / self.rx
        vy = (-y1_prime-cy_prime) / self.ry
        s = sign(ux * vy - uy * vx)
        delta_theta = s * py_ang([ux,uy],[vx,vy]) % 2 * pi
        theta_2 = theta_1 - delta_theta
        theta_max = max(theta_1,theta_2)
        theta_min = min(theta_1,theta_2)
        if not self.sweep:
            theta_delta -= 2*pi
        command = f"Calc.setExpression({{latex: '\\\left((-{self.ry})\\\sin\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\cos\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({c_x}),({self.ry})\\\cos\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\sin\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({c_y})\\\\right)', color: '{'#000000'}'}})"
        return command'''
        
    def arc(self, sx, sy, rx, ry, angle, large_arc, sweep, ex, ey):
        self.sx = sx 
        self.sy = sy
        self.rx = rx 
        self.ry = ry
        self.angle = angle 
        self.large_arc = large_arc
        self.sweep = not sweep 
        self.ex = ex 
        self.ey = ey
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
       
        #command = f"Calc.setExpression({{latex: '\\\left((-{self.ry})\\\sin\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\cos\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({c_x}),({self.ry})\\\cos\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\sin\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({theta_max})-({theta_min})\\\\right)t+({theta_min})\\\\right)+({c_y})\\\\right)', color: '{'#000000'}'}})"
        command = f"Calc.setExpression({{latex: '\\\left((-{self.ry})\\\sin\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\cos\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({c_x}),({self.ry})\\\cos\\\left(({self.angle})\\\\right)\\\sin\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({self.rx})\\\sin\\\left(({self.angle})\\\\right)\\\cos\\\left(\\\left(({max_min})\\\\right)t+({theta_min})\\\\right)+({c_y})\\\\right)', color: '{'#000000'}'}})"
        
        return command
        
