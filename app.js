// --- VETORES 3D (OPERAÇÕES MATEMÁTICAS BÁSICAS) ---
const vecDot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vecCross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
];
const vecSub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vecAdd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vecMult = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vecLen = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
const vecNorm = (a) => {
    const len = vecLen(a);
    return len < 1e-8 ? [0, 0, 0] : [a[0] / len, a[1] / len, a[2] / len];
};

// --- ESTRUTURA DE BUSCA ESPACIAL (GRID HASH 3D) ---
class SpatialHash {
    constructor(points, cellSize) {
        this.points = points;
        this.cellSize = cellSize;
        this.grid = {};
        this.build();
    }

    getKey(p) {
        const xi = Math.floor(p[0] / this.cellSize);
        const yi = Math.floor(p[1] / this.cellSize);
        const zi = Math.floor(p[2] / this.cellSize);
        return `${xi},${yi},${zi}`;
    }

    build() {
        for (let i = 0; i < this.points.length; i++) {
            const key = this.getKey(this.points[i]);
            if (!this.grid[key]) {
                this.grid[key] = [];
            }
            this.grid[key].push(i);
        }
    }

    queryRadius(q, r) {
        const indices = [];
        const rSq = r * r;
        const minX = Math.floor((q[0] - r) / this.cellSize);
        const maxX = Math.floor((q[0] + r) / this.cellSize);
        const minY = Math.floor((q[1] - r) / this.cellSize);
        const maxY = Math.floor((q[1] + r) / this.cellSize);
        const minZ = Math.floor((q[2] - r) / this.cellSize);
        const maxZ = Math.floor((q[2] + r) / this.cellSize);

        for (let x = minX; x <= maxX; x++) {
            for (let y = minY; y <= maxY; y++) {
                for (let z = minZ; z <= maxZ; z++) {
                    const key = `${x},${y},${z}`;
                    const cellPoints = this.grid[key];
                    if (cellPoints) {
                        for (let i = 0; i < cellPoints.length; i++) {
                            const idx = cellPoints[i];
                            const p = this.points[idx];
                            const dx = p[0] - q[0];
                            const dy = p[1] - q[1];
                            const dz = p[2] - q[2];
                            const distSq = dx * dx + dy * dy + dz * dz;
                            if (distSq <= rSq) {
                                indices.push(idx);
                            }
                        }
                    }
                }
            }
        }
        return indices;
    }
}

// --- ALGORITMO BALL PIVOTING (BPA) ---

function calcularCentroEsfera(p1, p2, p3, raio, normalRef) {
    const v1 = vecSub(p2, p1);
    const v2 = vecSub(p3, p1);
    const n = vecCross(v1, v2);
    const nLen = vecLen(n);
    if (nLen < 1e-6) return null;
    let nNorm = vecMult(n, 1 / nLen);

    if (vecDot(nNorm, normalRef) < 0) {
        nNorm = vecMult(nNorm, -1);
    }

    const detA = nLen;
    const v1Sq = vecDot(v1, v1);
    const v2Sq = vecDot(v2, v2);

    const term1 = vecMult(vecCross(v2, nNorm), 0.5 * v1Sq);
    const term2 = vecMult(vecCross(nNorm, v1), 0.5 * v2Sq);

    const cRel = vecMult(vecAdd(term1, term2), 1 / detA);
    const circuncentro = vecAdd(p1, cRel);
    const rCircSq = vecDot(cRel, cRel);

    if (rCircSq > raio * raio) return null;

    const h = Math.sqrt(raio * raio - rCircSq);
    const centroEsfera = vecAdd(circuncentro, vecMult(nNorm, h));
    return centroEsfera;
}

function pivotarAresta(i, j, kOposto, pontos, normais, hash, raioBola) {
    const p1 = pontos[i];
    const p2 = pontos[j];
    const arestaMeio = vecMult(vecAdd(p1, p2), 0.5);

    const dP1P2 = vecLen(vecSub(p2, p1));
    if (dP1P2 < 1e-6) return null;
    const vAresta = vecMult(vecSub(p2, p1), 1 / dP1P2);

    const centroAnt = calcularCentroEsfera(p1, p2, pontos[kOposto], raioBola, normais[i]);
    if (centroAnt === null) return null;

    let vCentroAnt = vecSub(centroAnt, arestaMeio);
    vCentroAnt = vecSub(vCentroAnt, vecMult(vAresta, vecDot(vCentroAnt, vAresta)));
    const lenVCentroAnt = vecLen(vCentroAnt);
    if (lenVCentroAnt < 1e-6) return null;
    const vCentroAntNorm = vecMult(vCentroAnt, 1 / lenVCentroAnt);

    let melhorPonto = null;
    let menorAngulo = 2 * Math.PI;
    let melhorCentro = null;

    const vizinhos = hash.queryRadius(arestaMeio, raioBola * 2.0);

    for (let c of vizinhos) {
        if (c === i || c === j || c === kOposto) continue;

        const centroNovo = calcularCentroEsfera(p1, p2, pontos[c], raioBola, normais[i]);
        if (centroNovo === null) continue;

        const intrusos = hash.queryRadius(centroNovo, raioBola - 1e-4);
        if (intrusos.length > 0) continue;

        let vNovo = vecSub(centroNovo, arestaMeio);
        vNovo = vecSub(vNovo, vecMult(vAresta, vecDot(vNovo, vAresta)));
        const lenVNovo = vecLen(vNovo);
        if (lenVNovo < 1e-6) continue;
        const vNovoNorm = vecMult(vNovo, 1 / lenVNovo);

        let cosAng = vecDot(vCentroAntNorm, vNovoNorm);
        if (cosAng > 1.0) cosAng = 1.0;
        if (cosAng < -1.0) cosAng = -1.0;

        let angulo = Math.acos(cosAng);

        const crossP = vecCross(vCentroAntNorm, vNovoNorm);
        if (vecDot(crossP, vAresta) < 0) {
            angulo = 2 * Math.PI - angulo;
        }

        if (angulo < menorAngulo) {
            menorAngulo = angulo;
            melhorPonto = c;
            melhorCentro = centroNovo;
        }
    }

    if (melhorPonto !== null) {
        return {
            novoPonto: melhorPonto,
            novoCentro: melhorCentro,
            angulo: menorAngulo
        };
    }
    return null;
}

function processarBPA(pontos, normais, raioBola) {
    const hash = new SpatialHash(pontos, raioBola * 2.0);
    const history = [];
    const triangulos = [];
    const arestasStatus = new Map(); // key: "minIdx,maxIdx" -> status (oposto / null)
    const filaFronteira = []; // array of [u, v]

    const edgeKey = (u, v) => u < v ? `${u},${v}` : `${v},${u}`;

    function logStep(type, data, msg) {
        history.push({
            type: type,
            triangles: [...triangulos],
            frontier: filaFronteira.map(edge => [...edge]),
            arestasStatus: new Map(arestasStatus),
            msg: msg,
            data: data
        });
    }

    // Início da busca de semente
    logStep('seed-search', { activeIdx: 0 }, "Iniciando busca por triângulo semente...");

    let semente = null;
    let sementeCentro = null;

    for (let i = 0; i < pontos.length; i++) {
        const vizinhosIdx = hash.queryRadius(pontos[i], raioBola * 2.0);
        if (vizinhosIdx.length < 3) continue;

        for (let j of vizinhosIdx) {
            if (i === j) continue;
            for (let k of vizinhosIdx) {
                if (k === i || k === j) continue;

                const centroBola = calcularCentroEsfera(pontos[i], pontos[j], pontos[k], raioBola, normais[i]);
                if (centroBola !== null) {
                    const colisoes = hash.queryRadius(centroBola, raioBola - 1e-4);
                    if (colisoes.length === 0) {
                        semente = [i, j, k];
                        sementeCentro = centroBola;
                        break;
                    }
                }
            }
            if (semente) break;
        }
        if (semente) break;

        if (i % 20 === 0 || i === pontos.length - 1) {
            logStep('seed-search', { activeIdx: i }, `Procurando semente no ponto ${i}...`);
        }
    }

    if (!semente) {
        logStep('done', null, "Falha: Nenhum triângulo semente encontrado. Ajuste o raio da bola.");
        return history;
    }

    triangulos.push(semente);
    logStep('seed-found', {
        tri: semente,
        center: sementeCentro
    }, `Semente encontrada nos pontos [${semente.join(', ')}]!`);

    // Inicializa a fronteira
    const arestasIniciais = [
        [semente[0], semente[1], semente[2]],
        [semente[1], semente[2], semente[0]],
        [semente[2], semente[0], semente[1]]
    ];

    for (let [u, v, op] of arestasIniciais) {
        const key = edgeKey(u, v);
        arestasStatus.set(key, op);
        filaFronteira.push([u, v]);
    }

    logStep('frontier-init', {
        frontier: filaFronteira.map(edge => [...edge])
    }, "Fronteira inicializada com as 3 arestas da semente.");

    let loopLimit = 3000;
    while (filaFronteira.length > 0 && loopLimit > 0) {
        loopLimit--;
        const [u, v] = filaFronteira.shift();
        const keyAtual = edgeKey(u, v);

        if (arestasStatus.get(keyAtual) === null) {
            continue;
        }

        const oposto = arestasStatus.get(keyAtual);
        const resPivot = pivotarAresta(u, v, oposto, pontos, normais, hash, raioBola);

        if (resPivot !== null) {
            const { novoPonto, novoCentro, angulo } = resPivot;
            const novoTri = [v, u, novoPonto];
            triangulos.push(novoTri);

            arestasStatus.set(keyAtual, null);

            const novasArestas = [
                [v, novoPonto, u],
                [novoPonto, u, v]
            ];

            let closedMsg = "";
            for (let [a, b, opNovo] of novasArestas) {
                const keyNova = edgeKey(a, b);

                if (arestasStatus.has(keyNova)) {
                    if (arestasStatus.get(keyNova) !== null) {
                        arestasStatus.set(keyNova, null);
                        closedMsg += ` Conexão fechada na aresta (${a}, ${b}).`;

                        const idx = filaFronteira.findIndex(edge => edgeKey(edge[0], edge[1]) === keyNova);
                        if (idx !== -1) {
                            filaFronteira.splice(idx, 1);
                        }
                    }
                } else {
                    arestasStatus.set(keyNova, opNovo);
                    filaFronteira.push([a, b]);
                }
            }

            logStep('pivot', {
                edge: [u, v],
                oposto: oposto,
                novoPonto: novoPonto,
                novoCentro: novoCentro,
                antCentro: calcularCentroEsfera(pontos[u], pontos[v], pontos[oposto], raioBola, normais[u]),
                angulo: angulo,
                tri: novoTri
            }, `Giro na aresta (${u}, ${v}) para ponto ${novoPonto} (Giro de ${(angulo * 180 / Math.PI).toFixed(1)}°).${closedMsg}`);

        } else {
            arestasStatus.set(keyAtual, null);
            logStep('boundary', {
                edge: [u, v]
            }, `Sem vizinhos para girar na aresta (${u}, ${v}). Definida como borda/vazia.`);
        }
    }

    logStep('done', null, `Reconstrução finalizada com sucesso! Total de triângulos gerados: ${triangulos.length}.`);
    return history;
}

// --- GERADORES DE NUVEM DE PONTOS ---

function gerarNuvem(tipo, n) {
    const pontos = [];
    const normais = [];

    if (tipo === 'hemisphere') {
        for (let i = 0; i < n; i++) {
            const theta = Math.random() * 2 * Math.PI;
            // Calota superior: cos(phi) varia de 0 a 1 -> phi varia de 0 a PI/2
            const cosPhi = Math.random();
            const sinPhi = Math.sqrt(1 - cosPhi * cosPhi);

            const x = sinPhi * Math.cos(theta);
            const y = sinPhi * Math.sin(theta);
            const z = cosPhi;

            pontos.push([x, y, z]);
            normais.push([x, y, z]); // Para esfera, normal é a própria posição normalizada
        }
    } else if (tipo === 'torus') {
        const R = 0.9; // Raio maior
        const r = 0.3; // Raio menor
        for (let i = 0; i < n; i++) {
            const theta = Math.random() * 2 * Math.PI;
            const phi = Math.random() * 2 * Math.PI;

            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);
            const cosPhi = Math.cos(phi);
            const sinPhi = Math.sin(phi);

            const x = (R + r * cosPhi) * cosTheta;
            const y = (R + r * cosPhi) * sinTheta;
            const z = r * sinPhi;

            // Centro do tubo para esse theta
            const cx = R * cosTheta;
            const cy = R * sinTheta;
            const cz = 0;

            // Vetor normal aponta do centro do tubo para o ponto
            const nx = (x - cx) / r;
            const ny = (y - cy) / r;
            const nz = (z - cz) / r;

            pontos.push([x, y, z]);
            normais.push([nx, ny, nz]);
        }
    } else if (tipo === 'wave') {
        for (let i = 0; i < n; i++) {
            const x = Math.random() * 2.0 - 1.0; // [-1, 1]
            const y = Math.random() * 2.0 - 1.0; // [-1, 1]
            const z = 0.25 * Math.sin(Math.PI * x) * Math.cos(Math.PI * y);

            // Derivadas parciais
            // dz/dx = 0.25 * pi * cos(pi * x) * cos(pi * y)
            // dz/dy = -0.25 * pi * sin(pi * x) * sin(pi * y)
            const dzdx = 0.25 * Math.PI * Math.cos(Math.PI * x) * Math.cos(Math.PI * y);
            const dzdy = -0.25 * Math.PI * Math.sin(Math.PI * x) * Math.sin(Math.PI * y);

            // Vetor normal: (-dzdx, -dzdy, 1) normalizado
            const nNorm = vecNorm([-dzdx, -dzdy, 1.0]);

            pontos.push([x, y, z]);
            normais.push(nNorm);
        }
    } else if (tipo === 'sphere') {
        for (let i = 0; i < n; i++) {
            const theta = Math.random() * 2 * Math.PI;
            const cosPhi = Math.random() * 2.0 - 1.0; // Esfera completa [-1, 1]
            const sinPhi = Math.sqrt(1 - cosPhi * cosPhi);

            const x = sinPhi * Math.cos(theta);
            const y = sinPhi * Math.sin(theta);
            const z = cosPhi;

            pontos.push([x, y, z]);
            normais.push([x, y, z]);
        }
    }

    // Adiciona uma pequena perturbação aleatória nos pontos para evitar casos colineares perfeitos de 3 pontos
    for (let i = 0; i < pontos.length; i++) {
        pontos[i][0] += (Math.random() - 0.5) * 0.005;
        pontos[i][1] += (Math.random() - 0.5) * 0.005;
        pontos[i][2] += (Math.random() - 0.5) * 0.005;
    }

    return { pontos, normais };
}

// --- VISUALIZAÇÃO THREE.JS ---

let scene, camera, renderer, controls;
let pointSystem, meshObject, meshEdges, frontierLines, pivotBallMesh, normalsHelperLines;
let animatingTriangleMesh;

// Elementos da UI do HTML
const shapeSelect = document.getElementById('shape-select');
const pointCountInput = document.getElementById('point-count');
const pointCountVal = document.getElementById('point-count-val');
const ballRadiusInput = document.getElementById('ball-radius');
const ballRadiusVal = document.getElementById('ball-radius-val');
const btnGenerate = document.getElementById('btn-generate');

const togglePoints = document.getElementById('toggle-points');
const toggleMesh = document.getElementById('toggle-mesh');
const toggleFrontier = document.getElementById('toggle-frontier');
const toggleBall = document.getElementById('toggle-ball');
const toggleNormals = document.getElementById('toggle-normals');
const toggleAutorotate = document.getElementById('toggle-autorotate');

const consoleLog = document.getElementById('console-log');
const statTriangles = document.getElementById('stat-triangles');
const statFrontiers = document.getElementById('stat-frontiers');
const statUnvisited = document.getElementById('stat-unvisited');
const statStatus = document.getElementById('stat-status');

const btnRestart = document.getElementById('btn-restart');
const btnPrev = document.getElementById('btn-prev');
const btnPlayPause = document.getElementById('btn-play-pause');
const btnNext = document.getElementById('btn-next');
const timelineCurrent = document.getElementById('timeline-current');
const timelineSlider = document.getElementById('timeline-slider');
const timelineTotal = document.getElementById('timeline-total');
const speedSlider = document.getElementById('speed-slider');
const speedVal = document.getElementById('speed-val');

// Estados globais de animação
let pontosGlobal = [];
let normaisGlobal = [];
let historyGlobal = [];
let activeStep = 0;
let interpT = 0; // Fração de interpolação entre o passo atual e o próximo [0, 1]
let animPlaying = false;
let animSpeed = 25; // Controla o incremento de interpT por frame
let lastTime = 0;

// Inicializa a cena 3D do Three.js
function init3D() {
    const container = document.getElementById('canvas-container');

    scene = new THREE.Scene();
    // Fundo azul escuro espacial degradê sutil
    scene.background = null; // Vamos usar o CSS background

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 50);
    camera.position.set(2, 2, 2.5);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 10;
    controls.minDistance = 0.5;

    // Iluminação premium e cinematográfica
    const ambientLight = new THREE.AmbientLight(0x0e1422, 1.5);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x00bcd4, 2.5); // Luz azul
    dirLight1.position.set(5, 8, 5);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x8a2be2, 2); // Luz roxa
    dirLight2.position.set(-5, -3, -5);
    scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0xffffff, 1, 10);
    pointLight.position.set(0, 2, 0);
    scene.add(pointLight);

    // Ajusta o tamanho no redimensionamento da janela
    window.addEventListener('resize', onWindowResize);

    setupSceneObjects();
}

function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Cria os materiais e objetos 3D vazios
function setupSceneObjects() {
    // 1. Nuvem de Pontos
    const pointsGeometry = new THREE.BufferGeometry();
    // Textura de bolinha brilhante desenhada dinamicamente em 2D canvas
    const pTexture = createCircleTexture('#00ffff');
    const pointsMaterial = new THREE.PointsMaterial({
        size: 0.045,
        map: pTexture,
        transparent: true,
        alphaTest: 0.1,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });
    pointSystem = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointSystem);

    // 2. Malha de Triângulos (Faces)
    const meshGeometry = new THREE.BufferGeometry();
    const meshMaterial = new THREE.MeshStandardMaterial({
        color: 0x00838f,
        roughness: 0.2,
        metalness: 0.3,
        transparent: true,
        opacity: 0.65,
        side: THREE.DoubleSide,
        flatShading: true
    });
    meshObject = new THREE.Mesh(meshGeometry, meshMaterial);
    scene.add(meshObject);

    // 3. Arestas da Malha (Wireframe sutil)
    const edgeMaterial = new THREE.LineBasicMaterial({
        color: 0x80deea,
        transparent: true,
        opacity: 0.25
    });
    meshEdges = new THREE.LineSegments(new THREE.BufferGeometry(), edgeMaterial);
    scene.add(meshEdges);

    // 4. Triângulo sendo animado (Fade in)
    const animTriGeometry = new THREE.BufferGeometry();
    const animTriMaterial = new THREE.MeshBasicMaterial({
        color: 0x00e5ff,
        transparent: true,
        opacity: 0.0,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    animatingTriangleMesh = new THREE.Mesh(animTriGeometry, animTriMaterial);
    scene.add(animatingTriangleMesh);

    // 5. Fronteira Ativa (Linhas verdes neon brilhantes)
    const frontierMaterial = new THREE.LineBasicMaterial({
        color: 0x39ff14,
        linewidth: 2, // Nota: linewidth > 1 não é suportado pela maioria dos drivers WebGL nativos, mas mantemos para fins semânticos
        transparent: true,
        opacity: 0.9
    });
    frontierLines = new THREE.LineSegments(new THREE.BufferGeometry(), frontierMaterial);
    scene.add(frontierLines);

    // 6. Bola de Pivô (Esfera de arame translúcida)
    const ballGeometry = new THREE.SphereGeometry(1, 32, 32);
    const ballMaterial = new THREE.MeshBasicMaterial({
        color: 0xffd700,
        wireframe: true,
        transparent: true,
        opacity: 0.2
    });
    pivotBallMesh = new THREE.Mesh(ballGeometry, ballMaterial);
    scene.add(pivotBallMesh);

    // Adiciona uma casca externa brilhante para a bola
    const ballCoreGeo = new THREE.SphereGeometry(0.015, 8, 8);
    const ballCoreMat = new THREE.MeshBasicMaterial({ color: 0xff3860 });
    const ballCore = new THREE.Mesh(ballCoreGeo, ballCoreMat);
    pivotBallMesh.add(ballCore);

    // 7. Vetores Normais
    const normalsMaterial = new THREE.LineBasicMaterial({
        color: 0xff00ff,
        transparent: true,
        opacity: 0.4
    });
    normalsHelperLines = new THREE.LineSegments(new THREE.BufferGeometry(), normalsMaterial);
    scene.add(normalsHelperLines);
}

// Auxiliar para criar textura circular suave
function createCircleTexture(colorStr) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 30);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.2, colorStr);
    grad.addColorStop(0.5, 'rgba(0, 188, 212, 0.4)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
}

// --- FUNÇÃO DE ATUALIZAÇÃO DA CENA POR PASSO/FRAME ---

function renderizarPasso(stepIdx, tInterp) {
    if (historyGlobal.length === 0 || stepIdx >= historyGlobal.length) return;

    const step = historyGlobal[stepIdx];
    const r = parseFloat(ballRadiusInput.value);

    // 1. Mostrar/Ocultar com base nos checkboxes
    pointSystem.visible = togglePoints.checked;
    meshObject.visible = toggleMesh.checked;
    meshEdges.visible = toggleMesh.checked;
    frontierLines.visible = toggleFrontier.checked;
    pivotBallMesh.visible = toggleBall.checked;
    normalsHelperLines.visible = toggleNormals.checked;

    // 2. Atualiza a malha consolidada (triângulos consolidados até o passo anterior)
    const meshGeometry = meshObject.geometry;
    const edgeGeometry = meshEdges.geometry;

    let triIndices = [];
    for (let i = 0; i < step.triangles.length; i++) {
        // Se for o passo de pivô atual, não adicionamos o último triângulo ainda na malha principal,
        // pois ele será exibido via animação de fade-in/giro
        if (step.type === 'pivot' && i === step.triangles.length - 1) {
            continue;
        }
        triIndices.push(step.triangles[i][0], step.triangles[i][1], step.triangles[i][2]);
    }

    meshGeometry.setIndex(triIndices);
    meshGeometry.attributes.position.needsUpdate = true;
    meshGeometry.computeVertexNormals();

    // Arestas da malha consolidada
    let edgeIndices = [];
    for (let i = 0; i < triIndices.length; i += 3) {
        const u = triIndices[i];
        const v = triIndices[i + 1];
        const w = triIndices[i + 2];
        edgeIndices.push(u, v, v, w, w, u);
    }
    edgeGeometry.setIndex(edgeIndices);
    edgeGeometry.attributes.position.needsUpdate = true;

    // 3. Atualiza as linhas da Fronteira Ativa
    const frontierGeometry = frontierLines.geometry;
    let frontIndices = [];
    for (let edge of step.frontier) {
        // Se a aresta estiver fechada no status, não mostramos
        const key = edge[0] < edge[1] ? `${edge[0]},${edge[1]}` : `${edge[1]},${edge[0]}`;
        if (step.arestasStatus.get(key) !== null) {
            frontIndices.push(edge[0], edge[1]);
        }
    }
    frontierGeometry.setIndex(frontIndices);
    frontierGeometry.attributes.position.needsUpdate = true;

    // 4. Animação da Bola de Pivô e do Triângulo novo
    animatingTriangleMesh.visible = false;

    if (step.type === 'seed-search') {
        // Bola passeia procurando semente
        pivotBallMesh.scale.set(r, r, r);
        const pIdx = step.data.activeIdx;
        const pos = pontosGlobal[pIdx];
        pivotBallMesh.position.set(pos[0], pos[1], pos[2]);

    } else if (step.type === 'seed-found') {
        // Bola pousa na semente inicial
        pivotBallMesh.scale.set(r, r, r);
        const c = step.data.center;
        pivotBallMesh.position.set(c[0], c[1], c[2]);

        // Exibe triângulo inicial com fade-in baseado no tInterp
        animatingTriangleMesh.visible = toggleMesh.checked;
        const tri = step.data.tri;
        const triGeo = animatingTriangleMesh.geometry;
        const p1 = pontosGlobal[tri[0]];
        const p2 = pontosGlobal[tri[1]];
        const p3 = pontosGlobal[tri[2]];

        const vertices = new Float32Array([...p1, ...p2, ...p3]);
        triGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        triGeo.computeVertexNormals();
        animatingTriangleMesh.material.color.setHex(0xff3860); // Vermelho semente
        animatingTriangleMesh.material.opacity = tInterp * 0.8;

    } else if (step.type === 'pivot') {
        pivotBallMesh.scale.set(r, r, r);

        const edge = step.data.edge;
        const u = edge[0];
        const v = edge[1];

        const p1 = pontosGlobal[u];
        const p2 = pontosGlobal[v];
        const pNovo = pontosGlobal[step.data.novoPonto];

        const cOld = step.data.antCentro;
        const cNew = step.data.novoCentro;

        const anguloMax = step.data.angulo;

        // Eixo de rotação (aresta de pivô)
        const arestaMeio = vecMult(vecAdd(p1, p2), 0.5);
        const dP1P2 = vecLen(vecSub(p2, p1));
        const vAresta = vecMult(vecSub(p2, p1), 1 / dP1P2);

        // Vetor perpendicular até o centro antigo da bola
        let vStart = vecSub(cOld, arestaMeio);
        vStart = vecSub(vStart, vecMult(vAresta, vecDot(vStart, vAresta)));
        const rPivot = vecLen(vStart);
        const vStartNorm = vecMult(vStart, 1 / rPivot);

        // Vetor ortogonal no plano de rotação
        const vPerp = vecCross(vAresta, vStartNorm);

        // Roda a bola usando interpolação em arco
        const anguloAtual = tInterp * anguloMax;
        const cX = arestaMeio[0] + rPivot * (vStartNorm[0] * Math.cos(anguloAtual) + vPerp[0] * Math.sin(anguloAtual));
        const cY = arestaMeio[1] + rPivot * (vStartNorm[1] * Math.cos(anguloAtual) + vPerp[1] * Math.sin(anguloAtual));
        const cZ = arestaMeio[2] + rPivot * (vStartNorm[2] * Math.cos(anguloAtual) + vPerp[2] * Math.sin(anguloAtual));

        pivotBallMesh.position.set(cX, cY, cZ);

        // Anima o novo triângulo aparecendo suavemente
        animatingTriangleMesh.visible = toggleMesh.checked;
        const triGeo = animatingTriangleMesh.geometry;

        // A ordem dos vértices segue a orientação de construção
        const vertices = new Float32Array([...p2, ...p1, ...pNovo]);
        triGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        triGeo.computeVertexNormals();
        animatingTriangleMesh.material.color.setHex(0x00e5ff); // Ciano neon normal
        animatingTriangleMesh.material.opacity = tInterp * 0.7;

    } else if (step.type === 'boundary') {
        // Bola pisca no centro da aresta sem saída
        const edge = step.data.edge;
        const p1 = pontosGlobal[edge[0]];
        const p2 = pontosGlobal[edge[1]];
        const arestaMeio = vecMult(vecAdd(p1, p2), 0.5);

        pivotBallMesh.position.set(arestaMeio[0], arestaMeio[1], arestaMeio[2]);
        pivotBallMesh.scale.set(r * 0.5, r * 0.5, r * 0.5); // Fica menor

    } else if (step.type === 'done') {
        // Bola encolhe e some no final
        const scale = Math.max(0, 1.0 - tInterp) * r;
        pivotBallMesh.scale.set(scale, scale, scale);
    }

    // 5. HUD de Estatísticas e Logs
    statTriangles.innerText = step.triangles.length;

    let boundaryCount = 0;
    step.arestasStatus.forEach((val) => {
        if (val === null) boundaryCount++;
    });
    // Quantidade real de arestas de fronteira ativas
    statFrontiers.innerText = step.frontier.length;

    // Calcula pontos visitados
    const visitados = new Set();
    for (let tri of step.triangles) {
        visitados.add(tri[0]);
        visitados.add(tri[1]);
        visitados.add(tri[2]);
    }
    const percentVisitados = ((visitados.size / pontosGlobal.length) * 100).toFixed(0);
    statUnvisited.innerText = `${100 - percentVisitados}%`;

    // Atualiza o texto de status
    if (step.type === 'seed-search') {
        statStatus.innerText = "Procurando Semente";
        statStatus.style.color = "var(--accent-purple)";
    } else if (step.type === 'seed-found') {
        statStatus.innerText = "Semente Encontrada";
        statStatus.style.color = "var(--accent-red)";
    } else if (step.type === 'pivot') {
        statStatus.innerText = "Pivoteando Bola";
        statStatus.style.color = "var(--primary)";
    } else if (step.type === 'boundary') {
        statStatus.innerText = "Aresta Limite";
        statStatus.style.color = "var(--accent-green)";
    } else if (step.type === 'done') {
        statStatus.innerText = "Concluído";
        statStatus.style.color = "var(--accent-gold)";
    }

    // Mantém o console log do passo visível
    atualizarConsole(stepIdx, step.msg);

    // Ajusta sliders e indicadores da timeline no HTML
    timelineCurrent.innerText = stepIdx;
    timelineSlider.value = stepIdx;
}

// Log box do console
let logsExibidos = new Set();

function atualizarConsole(stepIdx, msg) {
    const key = `${stepIdx}:${msg}`;

    // Se mudou de direção na timeline, limpa e reconstrói logs recentes para manter coerência
    if (stepIdx === 0) {
        consoleLog.innerHTML = `<div class="log-entry system">Sistema reiniciado.</div>`;
        logsExibidos.clear();
    }

    if (!logsExibidos.has(key)) {
        // Encontra o tipo de mensagem
        const step = historyGlobal[stepIdx];
        const div = document.createElement('div');
        div.className = `log-entry ${step.type}`;
        div.innerText = `[Passo ${stepIdx}] ${msg}`;
        consoleLog.appendChild(div);
        consoleLog.scrollTop = consoleLog.scrollHeight;
        logsExibidos.add(key);
    }
}

// --- CONTROLE DE ANIMAÇÃO / LOOP DE FLUXO ---

function animLoop(timestamp) {
    if (!lastTime) lastTime = timestamp;
    const dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    if (animPlaying) {
        // Incrementa t baseado na velocidade
        // Se speed = 25, o passo dura aproximadamente 1/25 segundos = 0.04s, então dt * velocidade
        const velocidadeFator = (animSpeed / 10) + 0.1;
        interpT += dt * velocidadeFator;

        if (interpT >= 1.0) {
            interpT = 0;
            activeStep++;
            if (activeStep >= historyGlobal.length) {
                activeStep = historyGlobal.length - 1;
                interpT = 1.0;
                animPlaying = false;
                setPlayState(false);
            }
        }
        renderizarPasso(activeStep, interpT);
    }

    // Auto-rotaciona a câmera se selecionado
    if (toggleAutorotate.checked) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
    } else {
        controls.autoRotate = false;
    }

    controls.update();
    renderer.render(scene, camera);

    requestAnimationFrame(animLoop);
}

function setPlayState(play) {
    animPlaying = play;
    if (play) {
        document.querySelector('.play-icon').classList.add('hidden');
        document.querySelector('.pause-icon').classList.remove('hidden');
    } else {
        document.querySelector('.play-icon').classList.remove('hidden');
        document.querySelector('.pause-icon').classList.add('hidden');
    }
}

// --- CRIAÇÃO DE DADOS E DISPARO ---

function inicializarBpaCompleto() {
    // Parar animação ativa
    setPlayState(false);

    const count = parseInt(pointCountInput.value);
    const tipo = shapeSelect.value;
    const r = parseFloat(ballRadiusInput.value);

    consoleLog.innerHTML = `<div class="log-entry system">Gerando nuvem de pontos (${tipo}, N = ${count})...</div>`;
    logsExibidos.clear();

    const nuvem = gerarNuvem(tipo, count);
    pontosGlobal = nuvem.pontos;
    normaisGlobal = nuvem.normais;

    // Atualiza a geometria da nuvem no Three.js
    const pointsGeo = pointSystem.geometry;
    const positions = new Float32Array(pontosGlobal.flat());
    pointsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointsGeo.computeBoundingSphere();

    // Centraliza e ajusta a câmera com base na nuvem de pontos gerada
    const center = pointsGeo.boundingSphere.center;
    controls.target.copy(center);

    // Atualiza geometria do mesh principal vazio inicialmente
    const meshGeo = meshObject.geometry;
    meshGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const edgeGeo = meshEdges.geometry;
    edgeGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const frontierGeo = frontierLines.geometry;
    frontierGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Atualiza linhas de normais
    const normalsGeo = normalsHelperLines.geometry;
    const normalsLinesPositions = [];
    for (let i = 0; i < pontosGlobal.length; i++) {
        const p = pontosGlobal[i];
        const n = normaisGlobal[i];
        // Adiciona linha do ponto p até p + n * 0.05
        normalsLinesPositions.push(p[0], p[1], p[2]);
        normalsLinesPositions.push(p[0] + n[0] * 0.06, p[1] + n[1] * 0.06, p[2] + n[2] * 0.06);
    }
    normalsGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(normalsLinesPositions), 3));

    // Roda BPA e salva o histórico
    consoleLog.innerHTML += `<div class="log-entry system">Processando algoritmo Ball Pivoting...</div>`;
    historyGlobal = processarBPA(pontosGlobal, normaisGlobal, r);

    // Configura os sliders de timeline
    activeStep = 0;
    interpT = 0;
    timelineSlider.max = historyGlobal.length - 1;
    timelineSlider.value = 0;
    timelineTotal.innerText = historyGlobal.length - 1;
    timelineCurrent.innerText = 0;

    // Roda render do primeiro passo
    renderizarPasso(0, 0);

    // Autoplay inicial
    setTimeout(() => {
        setPlayState(true);
    }, 500);
}

// --- VINCULAÇÃO DOS EVENT LISTENERS DA UI ---

function setupUIEventListeners() {
    // Sincroniza valores dos sliders nos labels
    pointCountInput.addEventListener('input', () => {
        pointCountVal.innerText = pointCountInput.value;
    });

    ballRadiusInput.addEventListener('input', () => {
        ballRadiusVal.innerText = ballRadiusInput.value;
    });

    btnGenerate.addEventListener('click', () => {
        inicializarBpaCompleto();
    });

    // Controles de Playback
    btnPlayPause.addEventListener('click', () => {
        if (historyGlobal.length === 0) return;
        setPlayState(!animPlaying);
    });

    btnRestart.addEventListener('click', () => {
        activeStep = 0;
        interpT = 0;
        setPlayState(false);
        logsExibidos.clear();
        consoleLog.innerHTML = `<div class="log-entry system">Simulação reiniciada pelo usuário.</div>`;
        renderizarPasso(0, 0);
    });

    btnNext.addEventListener('click', () => {
        if (historyGlobal.length === 0) return;
        setPlayState(false);
        if (activeStep < historyGlobal.length - 1) {
            activeStep++;
            interpT = 0;
            renderizarPasso(activeStep, 0);
        }
    });

    btnPrev.addEventListener('click', () => {
        if (historyGlobal.length === 0) return;
        setPlayState(false);
        if (activeStep > 0) {
            activeStep--;
            interpT = 0;
            // Para atualizar o console log retroativamente, precisamos resetar e recalcular os logs mostrados
            reconstruirLogsAte(activeStep);
            renderizarPasso(activeStep, 0);
        }
    });

    timelineSlider.addEventListener('input', () => {
        if (historyGlobal.length === 0) return;
        setPlayState(false);
        activeStep = parseInt(timelineSlider.value);
        interpT = 0;
        reconstruirLogsAte(activeStep);
        renderizarPasso(activeStep, 0);
    });

    speedSlider.addEventListener('input', () => {
        animSpeed = parseInt(speedSlider.value);
        speedVal.innerText = `${animSpeed}%`;
    });

    // Toggles de visualização disparam render imediato
    const updateRenderToggles = () => {
        if (historyGlobal.length > 0) {
            renderizarPasso(activeStep, interpT);
        }
    };

    togglePoints.addEventListener('change', updateRenderToggles);
    toggleMesh.addEventListener('change', updateRenderToggles);
    toggleFrontier.addEventListener('change', updateRenderToggles);
    toggleBall.addEventListener('change', updateRenderToggles);
    toggleNormals.addEventListener('change', updateRenderToggles);
}

// Reconstrói o console log até um determinado passo
function reconstruirLogsAte(stepLimit) {
    consoleLog.innerHTML = "";
    logsExibidos.clear();
    for (let i = 0; i <= stepLimit; i++) {
        atualizarConsole(i, historyGlobal[i].msg);
    }
}

// --- INICIALIZAÇÃO DA PÁGINA ---

window.addEventListener('load', () => {
    init3D();
    setupUIEventListeners();
    inicializarBpaCompleto(); // Inicialização automática do primeiro exemplo
    requestAnimationFrame(animLoop);
});
