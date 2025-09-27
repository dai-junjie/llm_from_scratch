#!/usr/bin/env python3
from pathlib import Path
from file import AgentFileReader, create_agent_file_reader


class SimpleAgent:
    def __init__(self, workspace_path: str = None):
        self.file_reader = create_agent_file_reader(workspace_path)
        self.workspace = Path(workspace_path) if workspace_path else Path.cwd()
        self.knowledge_base = {}
        
    def scan_project(self, extensions=['py', 'md', 'txt', 'json', 'yaml']):
        print(f"🔍 扫描项目: {self.workspace}")
        
        files = self.file_reader.list_by_extension(self.workspace, extensions, recursive=True)
        
        print(f"📁 发现 {len(files)} 个文件:")
        file_stats = {}
        for file_path in files:
            ext = file_path.suffix.lstrip('.')
            file_stats[ext] = file_stats.get(ext, 0) + 1
            
        for ext, count in sorted(file_stats.items()):
            print(f"  - .{ext}: {count} 个文件")
            
        return files
        
    def analyze_file(self, file_path):
        print(f"\n📄 分析文件: {file_path}")
        
        info = self.file_reader.get_file_info(file_path)
        if not info:
            print("❌ 无法获取文件信息")
            return None
            
        print(f"  📊 大小: {info['size']} 字节")
        print(f"  📝 类型: {info['mime_type'] or '未知'}")
        
        if info['suffix'] in ['.txt', '.md', '.py']:
            content = self.file_reader.read_text(file_path)
            if content:
                lines = len(content.splitlines())
                words = len(content.split())
                print(f"  📖 内容: {lines} 行, {words} 词")
                
                # 存储到知识库
                self.knowledge_base[str(file_path)] = {
                    'content': content,
                    'lines': lines,
                    'words': words,
                    'info': info
                }
                
        return info
        
    def search_content(self, query, file_extensions=['py', 'md', 'txt']):
        print(f"\n🔎 搜索内容: '{query}'")
        
        results = self.file_reader.find_files_by_content(
            self.workspace, query, file_extensions
        )
        
        if results:
            print(f"✅ 找到 {len(results)} 个匹配文件:")
            for result in results:
                file_name = Path(result['file']).name
                print(f"  - {file_name}: {result['matches']} 次匹配")
        else:
            print("❌ 未找到匹配内容")
            
        return results
        
    def get_project_summary(self):
        print(f"\n📋 项目总结:")
        
        tree = self.file_reader.read_directory_tree(self.workspace, max_depth=2)
        
        def count_items(node):
            count = 1 if not node.get('is_dir', False) else 0
            for child in node.get('children', []):
                count += count_items(child)
            return count
            
        total_files = count_items(tree)
        print(f"  📂 目录结构深度: 2 层")
        print(f"  📄 总文件数: {total_files}")
        print(f"  🧠 知识库条目: {len(self.knowledge_base)}")
        
        return tree
        
    def find_similar_files(self, target_file):
        print(f"\n🔗 寻找与 {target_file} 相似的文件:")
        
        if str(target_file) not in self.knowledge_base:
            print("❌ 目标文件不在知识库中")
            return []
            
        target_info = self.knowledge_base[str(target_file)]
        target_content = target_info['content'].lower()
        target_words = set(target_content.split())
        
        similar_files = []
        
        for file_path, info in self.knowledge_base.items():
            if file_path == str(target_file):
                continue
                
            content = info['content'].lower()
            words = set(content.split())
            
            # 计算词汇重叠度
            overlap = len(target_words & words)
            total_words = len(target_words | words)
            
            if total_words > 0:
                similarity = overlap / total_words
                if similarity > 0.1:  # 10%以上相似度
                    similar_files.append({
                        'file': file_path,
                        'similarity': similarity,
                        'common_words': overlap
                    })
                    
        similar_files.sort(key=lambda x: x['similarity'], reverse=True)
        
        if similar_files:
            print(f"✅ 找到 {len(similar_files)} 个相似文件:")
            for item in similar_files[:5]:  # 显示前5个
                file_name = Path(item['file']).name
                print(f"  - {file_name}: {item['similarity']:.2%} 相似度")
        else:
            print("❌ 未找到相似文件")
            
        return similar_files
        
    def interactive_mode(self):
        print("\n🤖 Agent交互模式启动!")
        print("命令: scan, analyze <file>, search <query>, summary, similar <file>, quit")
        
        while True:
            try:
                command = input("\n🤖 > ").strip()
                
                if command == "quit":
                    print("👋 再见!")
                    break
                elif command == "scan":
                    self.scan_project()
                elif command == "summary":
                    self.get_project_summary()
                elif command.startswith("search "):
                    query = command[7:]
                    self.search_content(query)
                elif command.startswith("analyze "):
                    file_path = command[8:]
                    self.analyze_file(file_path)
                elif command.startswith("similar "):
                    file_path = command[8:]
                    self.find_similar_files(file_path)
                else:
                    print("❓ 未知命令. 可用命令: scan, analyze, search, summary, similar, quit")
                    
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


def test_agent_functionality():
    print("🧪 Agent功能测试开始...\n")
    
    # 创建agent实例
    current_dir = Path(__file__).parent.parent.parent
    agent = SimpleAgent(str(current_dir))
    
    print("1️⃣ 测试项目扫描")
    files = agent.scan_project(['py'])
    
    print("\n2️⃣ 测试文件分析")
    if files:
        # 分析第一个Python文件
        agent.analyze_file(files[0])
    
    print("\n3️⃣ 测试内容搜索")
    agent.search_content("import", ['py'])
    
    print("\n4️⃣ 测试项目总结")
    agent.get_project_summary()
    
    print("\n5️⃣ 测试相似文件查找")
    if files:
        agent.find_similar_files(files[0])
    
    print("\n✅ Agent功能测试完成!")
    
    # 提供交互模式选项
    choice = input("\n💡 是否启动交互模式? (y/n): ").strip().lower()
    if choice == 'y':
        agent.interactive_mode()


if __name__ == "__main__":
    test_agent_functionality()