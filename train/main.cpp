#pragma GCC optimize(3,"Ofast","inline")
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <thread>
#include <mutex>
#include <windows.h>
#include "darknet.h"
const short AI = 1; // AI的棋子
const short OP = -1; // 对手的棋子
const short BLK = 0; // 空白

std::mutex mtx;
static omp_lock_t lock;

struct TrainData
{
    float input[4][81] = { 0 };
    float output[81] = { 0 };
};
std::vector<TrainData> trainData;

class AlphaPig
{
public:
    short board[81] = { 0 };
    bool air_vis[81];
    int mcts_cnt = 0, pl_cnt = 0;
    network* net;

    // 蒙特卡洛树节点
    struct treeNode
    {
        // 棋盘及颜色
        short board[81] = { 0 };
        short color;

        // 下一步可行位置
        short available_me[81], available_his[81];
        int available_me_size = 0, available_his_size = 0;
        bool available_me_map[81] = { false }, available_his_map[81] = { false };

        // 节点输赢统计
        double value = 0;
        int total = 0;
        int win = 0;
        bool fail = false;

        // 父节点
        treeNode* father = NULL;

        // 孩子节点
        treeNode* children[81];
        int children_size = 0;

        // 孩子探索策略
        short policy[81];
        int policy_size = 0;

        // 分支遍历完成数量
        int complete = 0;

        // 最后一手位置
        short last_p = -1;

        // 深度
        int depth = 0;
    };

    treeNode* root = NULL;

    // 策略权重
    struct Weight
    {
        short p;
        float w;
    };

    // 权重排序
    static bool weightCMP(const Weight& a, const  Weight& b)
    {
        return a.w > b.w;
    }

    // 移动位置
    inline short moveTo(short p, short dir)
    {
        switch (dir)
        {
        case 0:
            return (p += 9) < 81 ? p : -1;
        case 1:
            return (p -= 9) >= 0 ? p : -1;
        case 2:
            return p % 9 < 8 ? p + 1 : -1;
        case 3:
            return p % 9 > 0 ? p - 1 : -1;
        }
        return p;
    }

    // 判断是否有气
    bool hasAir(short mBoard[], short p)
    {
        air_vis[p] = true;
        bool flag = false;
        for (short dir = 0; dir < 4; dir++)
        {
            short dp = moveTo(p, dir);
            if (dp >= 0)
            {
                if (mBoard[dp] == BLK)
                    flag = true;
                if (mBoard[dp] == mBoard[p] && !air_vis[dp])
                    if (hasAir(mBoard, dp))
                        flag = true;
            }
        }
        return flag;
    }

    // 判断是否可以下子
    bool judgeAvailable(short mBoard[], short p, short col)
    {
        if (mBoard[p]) return false;
        mBoard[p] = col;
        memset(air_vis, 0, sizeof(air_vis));
        if (!hasAir(mBoard, p))
        {
            mBoard[p] = 0;
            return false;
        }
        for (short dir = 0; dir < 4; dir++)
        {
            short dp = moveTo(p, dir);
            if (dp >= 0)
            {
                if (mBoard[dp] && !air_vis[dp])
                    if (!hasAir(mBoard, dp))
                    {
                        mBoard[p] = 0;
                        return false;
                    }
            }
        }
        mBoard[p] = 0;
        return true;
    }

    // 扫描可以下子的位置
    void scanAvailable(treeNode* node)
    {
        short* board = node->board;
        bool ban_his[81] = { false }, ban_me[81] = { false }; // 禁下
        bool vis[81] = { false };

        for (short dir = 0; dir < 4; dir++)
        {
            short p = moveTo(node->last_p, dir);
            if(p == -1) continue;
            if (board[p] == BLK)
            {
                ban_me[p] = !judgeAvailable(board, p, node->color);
                ban_his[p] = !judgeAvailable(board, p, -node->color);
            }
            else if (!vis[p])
            {
                short queue[81], q_left = 0, q_right = 0;
                bool tgas_vis[81] = { false };
                short tgas = 0;
                int tgas_size = 0;
                queue[q_right++] = p;
                while (q_left != q_right)
                {
                    short pq = queue[q_left++];
                    q_left %= 81;
                    vis[pq] = true;
                    for (short dir = 0; dir < 4; dir++)
                    {
                        short dp = moveTo(pq, dir);
                        if (dp >= 0)
                        {
                            if (board[dp] == BLK && !tgas_vis[dp])
                            {
                                tgas_vis[dp] = true;
                                tgas_size++;
                                tgas = dp;
                            }
                            else if (board[dp] == board[pq] && !vis[dp])
                            {
                                queue[q_right++] = dp;
                                q_right %= 81;
                            }
                        }
                    }
                }
                if (tgas_size == 1)
                {
                    ban_me[tgas] = !judgeAvailable(board, tgas, node->color);
                    ban_his[tgas] = !judgeAvailable(board, tgas, -node->color);
                }
            }
        }

        for (int i = 0; i < node->father->available_me_size; i++)
        {
            short p = node->father->available_me[i];
            if (board[p] == BLK && !ban_his[p])
            {
                bool flag = true;
                for (short dir = 0; dir < 4; dir++)
                {
                    short dp = moveTo(p, dir);
                    if (dp >= 0 && board[dp] != node->color)
                    {
                        node->available_his[(node->available_his_size)++] = p;
                        node->available_his_map[p] = true;
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < node->father->available_his_size; i++)
        {
            short p = node->father->available_his[i];
            if (board[p] == BLK && !ban_me[p])
            {
                for (short dir = 0; dir < 4; dir++)
                {
                    short dp = moveTo(p, dir);
                    if (dp >= 0 && board[dp] != -node->color)
                    {
                        node->available_me[(node->available_me_size)++] = p;
                        node->available_me_map[p] = true;
                        break;
                    }
                }
            }
        }

    }

    // 策略函数
    void makePolicy(treeNode* node)
    {
        // 优先不走眼
        short eye[81] = { 0 }, no_eye[81] = { 0 };
        int eye_size = 0, no_eye_size = 0;
        short col = -node->color;

        for (int i = 0; i < node->available_his_size; i++)
        {
            short p = node->available_his[i];
            bool is_eye = true;
            for (short dir = 0; dir < 4; dir++)
            {
                short dp = moveTo(p, dir);
                if (dp >= 0 && node->board[dp] != col)
                {
                    is_eye = false;
                    break;
                }
            }
            if (is_eye)
            {
                eye[eye_size++] = p;
            }
            else
            {
                no_eye[no_eye_size++] = p;
            }
        }

        // 只剩下眼，直接返回
        if (no_eye_size == 0)
        {
            memcpy(node->policy, eye, sizeof(node->policy));
            node->policy_size = eye_size;

        }
        else
        {
            memcpy(node->policy, no_eye, sizeof(node->policy));
            node->policy_size = no_eye_size;
        }

        // 打乱
        for (int i = node->policy_size - 1; i >= 0; i--)
            std::swap(node->policy[i], node->policy[rand() % (i + 1)]);
    }

    // 估值函数
    inline double calcValue(treeNode* node)
    {
        // 暂时用可行下子位置估值
        double a = node->available_me_size;
        double b = node->available_his_size;
        if (a == 0 && b == 0 && node->father != NULL)
        {
            return -calcValue(node->father);
        }
        return 1 / (1 + pow(2.7182818284590452354, b - a)) * 2 - 1;
    }

    // 新建节点
    inline treeNode* newNode(treeNode* father, short p)
    {
        treeNode* newNode = new treeNode();
        memcpy(newNode->board, father->board, sizeof(board));
        newNode->color = -father->color;
        newNode->last_p = p;
        newNode->board[p] = newNode->color;
        newNode->father = father;
        newNode->depth = father->depth + 1;
        scanAvailable(newNode);
        makePolicy(newNode);
        father->children[father->children_size++] = newNode;
        return newNode;
    }

    // 删除分支
    void deleteTree(treeNode* node)
    {
        if (node != NULL)
        {
            while (node->children_size > 0)
                deleteTree(node->children[--node->children_size]);
            delete node;
        }
    }

    // 节点搜索完成
    inline bool finishNode(treeNode* node)
    {
        return (node->available_his_size > 0 && node->policy_size == 0 && node->complete == node->children_size) || (node->available_his_size == 0 && node->complete > 0);
    }

    // 选择最优子节点
    treeNode* bestChild(treeNode* node)
    {
        treeNode* max_node = NULL;
        bool Allcomplete = true;
        double max = -1e10;
        for (int i = 0; i < node->children_size; i++)
        {
            treeNode* t_node = node->children[i];
            if (finishNode(t_node))
                continue;

            // 上限置信区间算法
            double probability = t_node->value / t_node->total + 1.4142135623731 * sqrt(log(t_node->father->total) / t_node->total);
            if (probability > max)
            {
                max = probability;
                max_node = t_node;
                Allcomplete = false;
            }
        }
        return Allcomplete ? NULL : max_node;
    }

    // 选择&模拟&回溯
    bool select(treeNode* node)
    {
        // 选择
        while (node->available_his_size > 0) // 这个节点的游戏没有结束
        {
            if (node->policy_size > 0) // 这个节点有可行动作还未被拓展过
            {
                // 拓展
                node = newNode(node, node->policy[--node->policy_size]);
                break;
            }
            else   // 这个节点所有可行动作都被拓展过
            {
                node = bestChild(node);
                if (node == NULL)
                    return false;
            }
        }
        double value;

        // 模拟
        if (node->available_his_size == 0)   // 是否结束
        {
            node->complete = 1;
            treeNode* father = node->father;
            father->complete++;
            father->fail = true;
            while (father != NULL)
            {
                if (father->father == NULL)
                    break;
                if (finishNode(father))
                {
                    father->father->complete++;
                    if (father->fail == true)
                        father->father->win++;
                    if (father->win == father->complete)
                        father->father->fail = true;
                }
                father = father->father;
            }
            value = 1;
        }
        else
        {
            value = calcValue(node);
        }

        // 回溯
        while (node != NULL)
        {
            node->total += 1;
            node->value += value;
            node = node->father;
            value = -value;
        }

        return true;
    }

    // 初始化树
    void initRoot(short last_p)
    {
        root = new treeNode();
        memcpy(root, board, sizeof(board));
        root->color = OP;
        root->last_p = last_p;
        for (int i = 0; i < 81; i++)
        {
            if (judgeAvailable(root->board, i, root->color))
                root->available_me[(root->available_me_size)++] = i;
            if (judgeAvailable(root->board, i, -root->color))
                root->available_his[(root->available_his_size)++] = i;
        }
        makePolicy(root);
    }

    int choose(int last_p)
    {
        initRoot(last_p);

        // 我输了
        if (root->available_his_size == 0)
        {
            deleteTree(root);
            return -1;
        }

        // MCTS模拟
        for (int i = 0; i < 2000000 && select(root); i++);

        // 我输了
        if (finishNode(root) && root->win == root->complete)
        {
            deleteTree(root);
            return -1;
        }

        // 选择最好的下法
        treeNode* max_node = root->children[0];
        double max = -1e10;
        for (int i = 0; i < root->children_size; i++)
        {
            treeNode* t_node = root->children[i];
            double probability = t_node->value / t_node->total;
            if (probability > max)
            {
                max = probability;
                max_node = t_node;
            }
        }

        int ai = max_node->last_p;

        TrainData tdata;

        // 输入棋盘
        for (int i = 0; i < 81; i++)
        {
            if(board[i]==AI)
                tdata.input[0][i] = 1; // 第一通道为我方棋子
            if(board[i]==OP)
                tdata.input[1][i] = 1; // 第二通道为对方棋子
            if(board[i]==BLK)
                tdata.input[2][i] = 1; // 第三通道为空位
        }
        if(last_p>=0)
            tdata.input[3][last_p] = 1; // 第四通道为最后一手位置

        // 输出落子位置
        tdata.output[ai] = 1;

        // 加入训练集
        omp_set_lock(&lock);
        trainData.push_back(tdata);
        omp_unset_lock(&lock);

        deleteTree(root);

        return ai;
    }
};

void playGame(int i)
{
    while(true)
    {
        srand((unsigned)clock() + i);
        AlphaPig alphaPig1;
        AlphaPig alphaPig2;
        int a1 = -1, a2 = -1;
        while (true)
        {
            a1 = alphaPig1.choose(a2);
            if (a1 == -1)
            {
                break;
            }
            alphaPig1.board[a1] = AI;
            alphaPig2.board[a1] = OP;

            a2 = alphaPig2.choose(a1);
            if (a2 == -1)
            {
                break;
            }
            alphaPig1.board[a2] = OP;
            alphaPig2.board[a2] = AI;
        }
    }
}

int main()
{
    srand((unsigned)time(NULL));
    omp_init_lock(&lock);

    // 训练参数
    float avg_loss = -1;
    char* cfgfile = (char*)"policy_network.cfg";
    char* weightfile = (char*)"policy_network.weights";

    network* net = load_network(cfgfile, weightfile, false);
    if (net == NULL)
        return 0;

    size_t n = net->batch * net->subdivisions;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    // 记录开始时间
    double time = what_time_is_it_now();

    // 开始数据生成线程
    for(int i = 0; i < 32; i++)
    {
        std::thread t(playGame, i);
        t.detach();
    }

    // 等待数据
    while(true)
    {
        int length;
        omp_set_lock(&lock);
        length = trainData.size();
        omp_unset_lock(&lock);
        if(length > 30000)
            break;
        Sleep(1000);
    }

    // 开始训练
    while (get_current_batch(net) < net->max_batches || net->max_batches == 0)
    {
        omp_set_lock(&lock);
        size_t trainDataSize = trainData.size();
        data d = { 0 };
        d.X = make_matrix(n, 4 * 81);
        d.y = make_matrix(n, 81);
        for (size_t i = 0; i < n; i++)
        {
            TrainData tdata = trainData[rand() % trainDataSize];
            float* input = d.X.vals[i];
            float* output = d.y.vals[i];
            for (int j = 0; j < 81; j++)
            {
                input[0 * 81 + j] = tdata.input[0][j];
                input[1 * 81 + j] = tdata.input[1][j];
                input[2 * 81 + j] = tdata.input[2][j];
                input[3 * 81 + j] = tdata.input[3][j];
                output[j] = tdata.output[j];
            }

            int flip = rand() % 2;
            int rotate = rand() % 4;
            image in = float_to_image(9, 9, 4, input);
            image out = float_to_image(9, 9, 1, output);
            if (flip)
            {
                flip_image(in);
                flip_image(out);
            }
            rotate_image_cw(in, rotate);
            rotate_image_cw(out, rotate);
        }
        omp_unset_lock(&lock);

        float loss = train_network(net, d);
        free_data(d);

        if (avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss * .99 + loss * .01;

        if (get_current_batch(net) % 10 == 0)
        {
            printf("%d : %f, %f avg, %f rate, %lf seconds, %ld images, %ld total\n", (int)get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now() - time, (long)*net->seen, trainDataSize);
        }

        if (get_current_batch(net) % 1000 == 0)
        {
            save_weights(net, weightfile);
        }
    }

    free_network(net);
    return 0;
}
