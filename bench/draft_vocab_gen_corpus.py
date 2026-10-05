"""Generate a Chinese ranking corpus from the model itself (#196 follow-up).

The first CJK variant was ranked over ~670 tokens of hand-written text and landed
9% behind the full head, while a 705-token Cyrillic corpus reached parity. This
tests whether the corpus is the cause: same script, same --script-budget, only
the corpus differs.

Topics are deliberately disjoint from bench/prompts_han.jsonl (unicode/BPE, spring
nature, software performance, AI ethics, Renaissance painting, a photo-renamer
docstring, remote-work research) so the eval set stays held out from the ranking.
"""
import json, sys, time, urllib.request, argparse

# 48 topics, none of them an eval topic.
TOPICS = [
    "地质学与板块构造", "海洋学与洋流", "气象学与季风", "天文观测与望远镜", "生态学与生物多样性",
    "遗传学与基因编辑", "神经科学与大脑", "免疫学与疫苗", "流行病学与疾病传播", "营养学与膳食",
    "中世纪欧洲城堡建筑", "罗马帝国法律体系", "中国古代水利工程", "丝绸之路上的贸易", "伊斯兰几何艺术",
    "文艺复兴时期的音乐", "古典音乐曲式", "京剧唱腔与身段", "中国古典园林", "西方油画颜料史",
    "现代桥梁工程", "铁路与高速铁路", "民航发展史", "航海与造船", "电信网络演进",
    "计算机图形学", "密码学简史", "分布式系统", "数据库原理", "编译器设计",
    "操作系统内核", "计算机网络协议", "软件工程方法论", "人工智能伦理", "机器学习中的过拟合",
    "材料科学与合金", "超导材料", "半导体工艺", "核电站原理", "太阳能与风能",
    "农业机械与机械化", "茶叶栽培与加工", "酿酒工艺", "纺织工业史", "陶瓷烧制技术",
    "公共卫生政策", "城市规划与交通", "职业教育的变迁", "特殊教育的发展", "图书馆学与分类法",
]

def post(url, payload, key):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18020/v1/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--key", default="EMPTY")
    a = ap.parse_args()

    out = open(a.out, "w", encoding="utf-8")
    total = 0
    for i, topic in enumerate(TOPICS, 1):
        prompt = f"请写一篇大约八百字的中文文章，主题是{topic}。内容要准确、通顺、有知识性，使用小标题组织。"
        try:
            r = post(a.url, {"model": a.model, "prompt": prompt,
                             "max_tokens": a.max_tokens, "temperature": 0.7}, a.key)
            text = r["choices"][0]["text"]
        except Exception as e:
            print(f"  [{i:2}/{len(TOPICS)}] FAILED {topic}: {e}")
            continue
        out.write(text.strip() + "\n\n")
        out.flush()
        total += len(text)
        print(f"  [{i:2}/{len(TOPICS)}] {topic}: {len(text)} chars (total {total})")
    out.close()
    print(f"wrote {a.out}: {total} chars over {len(TOPICS)} topics")

if __name__ == "__main__":
    main()