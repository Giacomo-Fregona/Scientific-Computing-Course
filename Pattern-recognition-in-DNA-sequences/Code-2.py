#__________Scientific_Computing__________project2__________Giacomo_Fregona__________

def l2i(letter):#maps letters ACGT to values 0123 that can be used as array indexes
    if letter == 'A':
        return 0
    elif letter=='C':
        return 1
    elif letter =='G':
        return 2
    elif letter == 'T':
        return 3

class Tree: #class that we use to con struct a quaternary tree that represents of depth kk that will store subsequences of lenght kk
    global LL, kk, leaflist #list that stores the data (ex the genomic sequence), lenght of the mers (kk replaces k as variable name in the assignment), list of all the leafs (i.e. nodes of depth kk) of the tree
    def __init__(self, letter = '', dep = 0, fath = None):    #letter of the node, depth of the node, father of the node, frequency of the node
        self.letter = letter
        self.depth = dep
        self.father = fath
        self.frequency = 0
        self.childs = [None for _ in range(0,4)] #childs of the node
        self.firstvisittime=None #number that will be useful in Task 3 in order to retrieve the first position where we find the most frequent kmer

    def add(self,index):#adds to the three the sequence of length kk that starts in the position index of the list LL
        node=self
        for j in range(0,kk):
            l = LL[index+j]
            if node.childs[l2i(l)] == None:#if we are creating a new branch of the three
                for r in range(j,kk):
                    ll=LL[index+r]
                    node.childs[l2i(ll)] = Tree(ll,node.depth+1,node)
                    node=node.childs[l2i(ll)]
                break
            else :
                node = node.childs[l2i(l)]
        if node.frequency==0 : #update leaf frequency
            leaflist.append(node)
        node.frequency+=1

    def leaftokmer(self):#method that returns the kmer that corresponds to the given leaf
        currentnode=self
        l=""
        for i in range(0,kk):
            l=l+currentnode.letter
            currentnode = currentnode.father
        return l[::-1]

    def  fill(self,index):#method used in Task 3. Verifies if the kmer of length kk and starting in L[indes] has previously been added to the three. moreover it collects frequency and firstvisittime of the kmers
        if self.depth == kk:
            if self.firstvisittime==None :
                leaflist.append(self)
                self.firstvisittime=index-kk
                self.frequency=0
            self.frequency+=1
        else :
            l = LL[index]
            if self.childs[l2i(l)] != None:
                self.childs[l2i(l)].fill(index+1)

def frequencyofleaf(leaf):#service function used in order to apply the function max to the list leaflist (used below)
    return leaf.frequency
    
def kmer_hist(S, m):
    global LL,kk,leaflist
    LL = S
    kk=m
    leaflist=[]
    root = Tree()#create a treee where to store all the kmers we find
    for index in range(0,len(LL)-kk+1):#we add all the kmers of LL
        root.add(index)
    maxfreq = max(leaflist,key=frequencyofleaf).frequency#we find the most frequent kmer
    mfkmers = []
    h=[0 for i in range(0,maxfreq+1)]
    for index in range(0,len(leaflist)):#we construct the desired output
        f = leaflist[index].frequency
        if f == maxfreq :
            mfkmers.append(str(leaflist[index].leaftokmer()))
        h[f]+=1
    return h,mfkmers

def kmer_search(S, kmlist):
    global LL,kk,leaflist
    kk=len(kmlist[0])
    leaflist=[]
    root = Tree()#create a treee where to store all the kmers in the list kmlist
    for kmer in kmlist:#adds the kmers to the three
        LL = list(kmer)
        root.add(0)
    LL = S
    leaflist=[]
    for index in range(0,len(LL)-kk+1):#fills the tree with frequency and firstvisittime values for all the kmers that appears as subsequencies of LL
        root.fill(index)
    maxfreqnode = max(leaflist,key=frequencyofleaf,default=Tree())#finds the node that represents the most frequent kmer
    return maxfreqnode.firstvisittime,maxfreqnode.frequency