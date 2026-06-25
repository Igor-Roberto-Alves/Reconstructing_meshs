import numpy as np

class OctreeNode:
    """
    Representa um único nó (cubo) dentro da Octree.
    """
    def __init__(self, center, size, depth, max_depth):
        self.center = center         # Coordenada (x,y,z) do centro do nó
        self.size = size             # Tamanho da aresta do cubo
        self.depth = depth           # Profundidade atual na árvore (raiz = 0)
        self.max_depth = max_depth   # Profundidade máxima permitida
        
        self.is_leaf = True          # Todo nó nasce como folha
        self.children = []           # Lista de 8 filhos (se subdividido)
        
        # Estruturas para acumular dados reais da nuvem de pontos
        self.points = []
        self.normals = []
        
        # Propriedades acumuladas representativas deste nó
        self.avg_normal = np.zeros(3)
        self.center_of_mass = np.zeros(3)
        self.weight = 0              # Contagem de pontos neste nó

    def insert(self, point, normal):
        """
        Insere um ponto e sua normal no nó. Subdivide se necessário.
        """
        # Se for folha, mas ainda não atingiu o limite de profundidade, subdivide
        if self.is_leaf and self.depth < self.max_depth:
            self.subdivide()

        if self.is_leaf:
            # Se é folha (atingiu o max_depth), armazena os dados aqui
            self.points.append(point)
            self.normals.append(normal)
            
            # Atualiza o centro de massa e a normal média iterativamente
            self.weight += 1
            self.avg_normal += (normal - self.avg_normal) / self.weight
            self.center_of_mass += (point - self.center_of_mass) / self.weight
        else:
            # Se não é folha, repassa o ponto para o filho correto
            idx = self._get_octant(point)
            self.children[idx].insert(point, normal)

    def subdivide(self):
        """
        Divide este nó em 8 sub-cubos menores.
        """
        self.is_leaf = False
        quarter_size = self.size / 4.0
        half_size = self.size / 2.0
        
        for i in range(8):
            # Lógica de bits para descobrir o deslocamento em X, Y e Z
            # Ex: i=0 (000) -> -, -, - | i=7 (111) -> +, +, +
            offset_x = -quarter_size if (i & 1) == 0 else quarter_size
            offset_y = -quarter_size if (i & 2) == 0 else quarter_size
            offset_z = -quarter_size if (i & 4) == 0 else quarter_size
            
            child_center = self.center + np.array([offset_x, offset_y, offset_z])
            child = OctreeNode(child_center, half_size, self.depth + 1, self.max_depth)
            self.children.append(child)

    def _get_octant(self, point):
        """
        Retorna o índice (0 a 7) do quadrante no qual o ponto se encontra.
        """
        idx = 0
        if point[0] > self.center[0]: idx |= 1
        if point[1] > self.center[1]: idx |= 2
        if point[2] > self.center[2]: idx |= 4
        return idx

    def get_active_leaves(self, leaves_list):
        """
        Percorre a árvore e coleta todos os nós folhas que contêm pontos.
        """
        if self.is_leaf:
            if self.weight > 0:
                leaves_list.append(self)
        else:
            for child in self.children:
                child.get_active_leaves(leaves_list)


class Octree:
    """
    Classe gerenciadora da Octree que calcula os limites espaciais.
    """
    def __init__(self, points, max_depth=8):
        self.max_depth = max_depth
        
        # 1. Encontrar os limites do espaço tridimensional (Bounding Box)
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        
        # Adicionar uma margem de segurança de 5% para que nenhum ponto caia na borda exata
        margin = (max_pt - min_pt) * 0.05
        min_pt -= margin
        max_pt += margin
        
        # A Octree precisa ser um cubo perfeito, então pegamos a maior dimensão
        self.size = np.max(max_pt - min_pt)
        self.center = (min_pt + max_pt) / 2.0
        
        # 2. Inicializar o nó raiz
        self.root = OctreeNode(self.center, self.size, depth=0, max_depth=self.max_depth)

    def insert_cloud(self, points, normals):
        """
        Insere múltiplos pontos e normais de uma vez.
        """
        for p, n in zip(points, normals):
            self.root.insert(p, n)

    def get_leaves(self):
        """
        Retorna uma lista com todos os nós folha ativos (que contêm dados).
        """
        leaves = []
        self.root.get_active_leaves(leaves)
        return leaves


# --- EXEMPLO DE USO ---
if __name__ == "__main__":
    # Gerar pontos aleatórios em uma esfera
    print("Gerando 5000 pontos em uma esfera...")
    phi = np.random.uniform(0, np.pi, 5000)
    theta = np.random.uniform(0, 2 * np.pi, 5000)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    pontos = np.vstack((x, y, z)).T
    normais = pontos.copy()

    # Inicializar a Octree
    max_depth = 4 # Equivale a um grid de 16x16x16
    print(f"Construindo Octree com profundidade máxima = {max_depth}...")
    
    arvore = Octree(pontos, max_depth=max_depth)
    arvore.insert_cloud(pontos, normais)
    
    # Obter os resultados
    folhas_ativas = arvore.get_leaves()
    print(f"Total de pontos inseridos: {len(pontos)}")
    print(f"Total de nós (caixas) ativas geradas: {len(folhas_ativas)}")
    
    # Analisando uma das folhas
    amostra = folhas_ativas[0]
    print(f"\nExemplo de um nó folha:")
    print(f"- Pontos dentro desta caixa: {amostra.weight}")
    print(f"- Centro de massa calculado: {amostra.center_of_mass}")
    print(f"- Normal média calculada: {amostra.avg_normal}")