from ete3 import Tree, TreeStyle, NodeStyle
from zss import simple_distance, Node, distance
# from newick import read as nread
import io
import re
    

pattern = 'a-zA-Z'
reg = re.compile(pattern)

try:
    from editdist import distance as strdist
except ImportError:
    def strdist(a, b):
        if a == b:
            return 0
        else:
            return 1

def weird_dist(A, B):
    return strdist(A, B)

class PhyloTree(Tree):
    
    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.name
    
    @staticmethod
    def zhang_shasha_distance(t1, t2):
        return simple_distance(t1, t2, get_children = PhyloTree.get_children, get_label = PhyloTree.get_label)

    @staticmethod
    # rename the names of nodes that are "" to the given label
    # rename_all: rename all node's name to the given label
    def rename_node(tree, level = 0, label = "J", rename_all = False):
        
        children = tree.children
        if children is not None:
            for c in tree.get_children(tree):
                    PhyloTree.rename_node(c, level = level+1, label = label, rename_all = rename_all)
        if rename_all:
            tree.name = label    
        elif tree.name == "":
#             print "here"
            tree.name = label 
            
            
#         print "tree.name", tree.name
        return 
    
    @staticmethod
    def getNodeNum(tree, num = 1):
        children = tree.children
        if children is not None:
            for c in tree.get_children(tree):
                num = PhyloTree.getNodeNum(c, num = num) + 1        
        return num
    
    def getLeafCount(self):
        count = 0
        for node in self.traverse():
            if len(node.children) == 0:
                count += 1
        return count
    
#     @staticmethod
    def drawTree(self):
        ts = TreeStyle()
        ts.show_leaf_name = True
    #     ts.margin_right = 20
        ts.branch_vertical_margin = 10
    #     ts.branch_horizontal_margin = 10
        
        # Draws nodes as small red spheres of diameter equal to 10 pixels
        nstyle = NodeStyle()
        nstyle["shape"] = "circle"
        nstyle["size"] = 1
        
        # Gray dashed branch lines
    #     nstyle["hz_line_type"] = 2
    #     nstyle["hz_line_color"] = "#cccccc"
        
        # Applies the same static style to all nodes in the tree. Note that,
        # if "nstyle" is modified, changes will affect to all nodes
        for n in self.traverse():
            n.set_style(nstyle)
    
    
        self.show(tree_style=ts)
        
#     @staticmethod
#     read

    
def getElement(str, index):
    
    element = str[index]
#     print element
    is_alpha = element.isalpha()
    index += 1
    keep_going = True
    
    while keep_going and index < len(str):
        temp = str[index]
        if temp.isalpha() == is_alpha:
            element += temp
            index += 1
        else:
            keep_going = False
            
    return element, index
            
            
def getNextAlphabet(string):
    
#     print ord("Z")
    ord_list = []
    length = len(string)
    for char in string:
#         print char
#         print ord("Z")
        ord_list.append(ord(char))
        
    if ord_list[-1] < 90:
        ord_list[-1] += 1
    else:
        modified = False
        i = length - 2
        while i > -1 and not modified:
            if ord_list[i] < 90:
                ord_list[i] += 1
                for index in range(i+1, length):
                    ord_list[index] = 65
                modified = True
            
            i -= 1
        
        if not modified:
            temp = ["A"]*(length+1)
            return "".join(temp)
            
    new_string = ""
    for c in ord_list:
        new_string += chr(c)
    
    return new_string
            

## Convert rapid hand-label string to normal expression
## rename: rename all leaves' names to alphabet order
def string2TreeString(str, rename = False):
    stack_down = []
    stack_sibling = []

    index = 0
    length = len(str)
    # print "length", length
#     pattern = r'[a-zA-Z]'
    
    id = "A"
    while index < length:
#         print "index:", index
        name, index = getElement(str, index)
        
        if rename:
            name = id
            id = getNextAlphabet(id)

        
#         print "element: ", element 
#         print "next index:", index
        connect, index = getElement(str, index)
#         print "element: ", name, "symbol: ", connect

        repeat_count = 0 
        
        if connect == ",":
#             print","
    #         if len(stack_sibling) > 0:
    #             stack_down.append(stack_sibling.pop())
            substring = name
            stack_sibling.append(name)
       
        elif connect.isdigit():
            jump = int(connect)
            if jump == 0:
                stack_down.append(name)
            else:   
                while jump > 0:
                    if len(stack_sibling) > 0:
#                         print "merge sibling"
                        stack_sibling.append(name)
                        substring = "(" + ",".join(stack_sibling) + ")"
                        stack_sibling = []
                    else:
#                         print "up string, time: %d"%repeat_count
                        if repeat_count == 0:
#                             print "self jump"
                            substring = "(" + stack_down.pop() +","  + name + ")"
                        else:
#                             print "stack jump"
                            back = stack_down.pop()
                            front = stack_down.pop()
                            substring = "(" + front +","  + back + ")"
                     
                    jump -=1
                    repeat_count += 1
                     
                    stack_down.append(substring)
#                     print substring                  

        # non-binary handler
        else:
            jump = int(connect[0:-1])
            if jump == 0:
                stack_down.append(name)
            else:   
                while jump > 0:
                    if len(stack_sibling) > 0:
#                         print "merge sibling"
                        stack_sibling.append(name)
                        substring = "(" + ",".join(stack_sibling) + ")"
                        stack_sibling = []
                    else:
#                         print "up string, time: %d"%repeat_count
                        if repeat_count == 0:
#                             print "self jump"
                            substring = "(" + stack_down.pop() +","  + name + ")"
                        else:
#                             print "stack jump"
                            back = stack_down.pop()
                            front = stack_down.pop()
                            substring = "(" + front +","  + back + ")"
                     
                    jump -=1
                    repeat_count += 1
                     
                    stack_down.append(substring)
#                     print stack_down
            
            # handle non-binary case
#             print "handle non-binary tree"
#             print "connect", connect
            back = stack_down.pop() 
            front = stack_down.pop() 
#             print "back", back
#             print "front", front
            stack_down.append(front + "," + back)
#         print "stack_sibling", stack_sibling
#         print "stack_down", stack_down, len(stack_down)
         
    if len(stack_down) > 1:
        print "Tree is not complished! Please check your string."
        print stack_down
        return None
    else:
        return stack_down[0]




##########
def string2TreeStringOld(str):
    stack_down = []
    stack_sibling = []
 
    for i in range(0, len(str), 2):
#         print str[i]
#         print str[i+1]
        repeat_count = 0 
        name = str[i]
        connect = str[i+1]
       
        if connect == ",":
#             print","
    #         if len(stack_sibling) > 0:
    #             stack_down.append(stack_sibling.pop())
            substring = name
            stack_sibling.append(name)
       
        elif connect.isdigit():
            jump = int(connect)
            if jump == 0:
                stack_down.append(name)
            else:   
                while jump > 0:
                    if len(stack_sibling) > 0:
#                         print "merge sibling"
                        stack_sibling.append(name)
                        substring = "(" + ",".join(stack_sibling) + ")"
                        stack_sibling = []
    #                     jump -= 1
                    else:
#                         print "up string, time: %d"%repeat_count
                        if repeat_count == 0:
#                             print "self jump"
                            substring = "(" + stack_down.pop() +","  + name + ")"
                        else:
#                             print "stack jump"
                            back = stack_down.pop()
                            front = stack_down.pop()
                            substring = "(" + front +","  + back + ")"
    #                     jump -=1
                     
                    jump -=1
                    repeat_count += 1
                     
                    stack_down.append(substring)
#                     print substring
     
#         print "stack_sibling", stack_sibling
#         print "stack_down", stack_down
         
    if len(stack_down) > 1:
        print "Tree is not complished! Please check your string."
        print stack_down
        return None
    else:
        return stack_down[0]


def post_order(tree, level):
    
    children = tree.children
    if children is not None:
        for c in tree.get_children(tree):
#             if c.name is not None:
                post_order(c, level+1)

    print "%s, %d"%(tree.get_label(tree),level)
        
            




if __name__ == '__main__':
    
    
 

    t1 = PhyloTree("(1,((2,(3,(4,5))),(6,((((7,(8,9)),((10,11),(12,13))),(14,(((15,(16,(17,18))),(19,(20,21))),(22,23)))),(24,(((25,(26,27)),(28,(29,(30,31)))),((32,(33,(34,(35,(36,(37,38)))))),((39,40),(41,42)))))))));")
    print t1
    print t1.getLeafCount()
    
#     t1 = PhyloTree("(((((((Lafidae, Stercorariidae), Sternidae), Rynchopidae),Glareolidae,Burhinidae,Chionidae,(((Haematopodidae, Recurvirostridae), Vanellinae), Charadriinae)), ((((((Jacanidae, Rostratulidae), Gallinagininae), Tringinae), (Arenariinae, Calidrinae)), Phalaropodinae), Thinocoridae)), Alcidae), Outgroups);")
    
#     ts = TreeStyle()
#     ts.show_leaf_name = True
# #     ts.margin_right = 20
#     ts.branch_vertical_margin = 5
# #     ts.branch_horizontal_margin = 10
#     
#     # Draws nodes as small red spheres of diameter equal to 10 pixels
#     nstyle = NodeStyle()
#     nstyle["shape"] = "circle"
#     nstyle["size"] = 1
#     
#     # Gray dashed branch lines
# #     nstyle["hz_line_type"] = 2
# #     nstyle["hz_line_color"] = "#cccccc"
#     
#     # Applies the same static style to all nodes in the tree. Note that,
#     # if "nstyle" is modified, changes will affect to all nodes
#     for n in t1.traverse():
#         n.set_style(nstyle)


#     t1.show(tree_style=ts)

#     print PhyloTree.getNodeNum(t1)


#     t1 = PhyloTree("((a,(a,)),(d,(e,f)));")
#     t2 = PhyloTree("((a,b),(c,(d,e)));")
    
    
    
    

        

#     PhyloTree.rename_node(t1, rename_all=True)
#     PhyloTree.rename_node(t2, rename_all=True)
# 
#     print 
#     for n in t1.traverse():
#         print "name", n.name
#     
#     print 
#     for n in t2.traverse():
#         print "name", n.name
#         
#     print 
#         
#     print t1
# #     print PhyloTree.getNodeNum(t1)
#     print t2
# #     print PhyloTree.getNodeNum(t2)
# 
#     
#     print "distance", PhyloTree.zhang_shasha_distance(t1, t2)
    
#     answer =  t1.compare(t2, unrooted=True)
    
    # print t1
    # print t5
    # print answer
    # print "source: ", t1
    # print "ref: ", t2
    # print "effective size:", answer["effective_tree_size"]
    # print "nRF", answer["norm_rf"]
    # print "RF", answer["rf"]
    # print "maxRF", answer["max_rf"]
    # print "src_br", answer["source_edges_in_ref"]
    # print "ref_br", answer["ref_edges_in_source"], "common edge / valid ref edge = %d/%d"%(len(answer["common_edges"]),len(answer["ref_edges"]))
    
    # print Tree("((A,B), ((C,D),E));") 
    # print Tree("((C, D), (E, (A, B)));")
    
    

    #     print "%s, %d"%(tree.name,level)
        
    # t1 = Tree("((a,(b,c)),(d,(e,f)));")
    # print t1
    
#     t1 = PhyloTree("((X,(X,X)), (X,(X,X)));")
#     t2 = PhyloTree("((X,(X,X)), ((X,X),(X,X)));")
    
    
    
    # t2 = PhyloTree(" ((A, B), (C, (F, E)));")
#     print t1
#     print t2
    
    #     
#     A = (Node("J")
#             .addkid(Node("J")
#                 .addkid(Node("A"))
#                 .addkid(Node("J").addkid(Node("B"))
#                                 .addkid(Node("C"))))
#             .addkid(Node("J")
#                     .addkid(Node("D"))
#                     .addkid(Node("J").addkid(Node("E"))
#                             .   addkid(Node("F"))))
#         )
#     
#     
#     B = (Node("J")
#             .addkid(Node("J")
#                 .addkid(Node("A"))
#                 .addkid(Node("B")))
#             .addkid(Node("J")
#                     .addkid(Node("C"))
#                     .addkid(Node("J").addkid(Node("D"))
#                             .   addkid(Node("E"))))
#         )
#     # 
#     
# #     rename_node(t1, 0, label ="X")
# #     rename_node(t2, 0, label ="X")
#     print
#     print "t1"
#     print post_order(t1, 0)
#     print
#     print "A"
#     print post_order(A, 0)
# 
#     print "here", simple_distance(t1, t2, PhyloTree.get_children, PhyloTree.get_label)


