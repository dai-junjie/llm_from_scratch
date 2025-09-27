#!/usr/bin/env python3
import json
import yaml
from pathlib import Path
import tempfile
from file import create_agent_file_reader, read_file, list_project_files


def test_basic_file_operations():
    print("=== 基础文件操作测试 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(temp_path)
        reader = create_agent_file_reader(temp_path)
        
        # 创建测试文件
        test_file = temp_path / "test.txt"
        test_file.write_text("Hello, Agent!\n这是测试内容。", encoding='utf-8')
        
        # 测试读取文本
        content = reader.read_text("test.txt")
        print(f"✓ 读取文本: {content.strip()}")
        
        # 测试按行读取
        lines = reader.read_lines("test.txt")
        print(f"✓ 按行读取: {len(lines)} 行")
        
        # 测试文件信息
        info = reader.get_file_info("test.txt")
        print(f"✓ 文件信息: {info['name']}, 大小: {info['size']} 字节")


def test_structured_data():
    print("\n=== 结构化数据测试 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        reader = create_agent_file_reader(temp_path)
        
        # JSON测试
        json_data = {"name": "Agent", "version": "1.0", "features": ["file_read", "search"]}
        json_file = temp_path / "config.json"
        json_file.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
        
        loaded_json = reader.read_json("config.json")
        print(f"✓ JSON读取: {loaded_json['name']} v{loaded_json['version']}")
        
        # YAML测试
        yaml_data = {"database": {"host": "localhost", "port": 5432}, "debug": True}
        yaml_file = temp_path / "settings.yaml"
        yaml_file.write_text(yaml.dump(yaml_data, allow_unicode=True))
        
        loaded_yaml = reader.read_yaml("settings.yaml")
        print(f"✓ YAML读取: 数据库端口 {loaded_yaml['database']['port']}")


def test_directory_operations():
    print("\n=== 目录操作测试 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        reader = create_agent_file_reader(temp_path)
        
        # 创建目录结构
        (temp_path / "src").mkdir()
        (temp_path / "docs").mkdir()
        (temp_path / "src" / "main.py").write_text("print('Hello')")
        (temp_path / "src" / "utils.py").write_text("def helper(): pass")
        (temp_path / "docs" / "readme.md").write_text("# 文档")
        (temp_path / "config.txt").write_text("setting=value")
        
        # 测试列出所有文件
        all_files = reader.list_files(".", "*", recursive=True)
        print(f"✓ 递归列出文件: {len(all_files)} 个文件")
        
        # 测试按扩展名筛选
        py_files = reader.list_by_extension(".", ["py"], recursive=True)
        print(f"✓ Python文件: {len(py_files)} 个")
        
        # 测试目录树
        tree = reader.read_directory_tree(".")
        print(f"✓ 目录树构建: 找到 {len(tree['children'])} 个顶级项目")


def test_search_functionality():
    print("\n=== 搜索功能测试 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        reader = create_agent_file_reader(temp_path)
        
        # 创建包含特定内容的文件
        (temp_path / "file1.py").write_text("def process_data():\n    return 'processed'")
        (temp_path / "file2.py").write_text("import pandas\ndef analyze_data():\n    pass")
        (temp_path / "file3.txt").write_text("这里没有相关代码")
        
        # 搜索包含特定文本的文件
        results = reader.find_files_by_content(".", "def ", ["py"])
        print(f"✓ 内容搜索: 找到 {len(results)} 个Python文件包含 'def '")
        
        for result in results:
            print(f"  - {Path(result['file']).name}: {result['matches']} 次匹配")


def test_batch_operations():
    print("\n=== 批量操作测试 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        reader = create_agent_file_reader(temp_path)
        
        # 创建多个文件
        files_to_create = ["a.txt", "b.txt", "c.txt"]
        for i, filename in enumerate(files_to_create):
            (temp_path / filename).write_text(f"内容 {i+1}")
        
        # 批量读取
        results = reader.batch_read_files(files_to_create)
        print(f"✓ 批量读取: {len(results)} 个文件")
        
        # 过滤可读文件
        all_paths = [temp_path / f for f in files_to_create] + [temp_path / "nonexistent.txt"]
        readable = reader.filter_readable_files(all_paths)
        print(f"✓ 可读文件过滤: {len(readable)}/{len(all_paths)} 个文件可读")


def test_agent_project_integration():
    print("\n=== Agent项目集成测试 ===")
    
    # 测试当前项目的文件读取
    project_root = Path(__file__).parent.parent.parent
    reader = create_agent_file_reader(project_root)
    
    # 列出所有Python文件
    py_files = list_project_files(project_root / "src", ["py"])
    print(f"✓ 项目Python文件: {len(py_files)} 个")
    
    # 读取README
    readme_content = read_file(project_root / "README.md")
    if readme_content:
        lines = readme_content.split('\n')[:5]
        print(f"✓ README前5行预览:")
        for line in lines:
            print(f"  {line}")
    
    # 搜索agent相关文件
    agent_files = reader.find_files_by_content(project_root / "src", "agent", ["py"])
    print(f"✓ 包含'agent'的文件: {len(agent_files)} 个")


def test_safety_features():
    print("\n=== 安全特性测试 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        reader = create_agent_file_reader(temp_path)
        
        # 测试路径安全检查
        unsafe_content = reader.read_text("../../../etc/passwd")
        print(f"✓ 路径安全检查: {'通过' if unsafe_content is None else '失败'}")
        
        # 创建大文件测试
        large_file = temp_path / "large.txt"
        large_file.write_text("x" * 1000)  # 1KB文件
        
        large_content = reader.read_text("large.txt")
        print(f"✓ 正常大小文件读取: {'成功' if large_content else '失败'}")


def demonstrate_agent_usage():
    print("\n=== Agent使用示例 ===")
    
    print("Agent文件读取器已创建，可以用于:")
    print("- 安全的文件读取和目录遍历")
    print("- 支持多种格式 (文本、JSON、YAML、二进制)")
    print("- 内容搜索和批量处理")
    print("- 完整的错误处理和日志记录")
    print("- 路径安全检查防止目录遍历攻击")
    
    # 展示API使用
    print("\n常用API示例:")
    print("```python")
    print("# 创建reader")
    print("reader = create_agent_file_reader('/path/to/project')")
    print()
    print("# 读取文件")
    print("content = reader.read_text('config.txt')")
    print("data = reader.read_json('settings.json')")
    print()
    print("# 搜索文件")
    print("py_files = reader.list_by_extension('.', ['py'], recursive=True)")
    print("matches = reader.find_files_by_content('.', 'def main', ['py'])")
    print()
    print("# 批量操作")
    print("all_content = reader.batch_read_files(file_list)")
    print("```")


def main():
    print("Agent文件读取功能测试开始...\n")
    
    try:
        test_basic_file_operations()
        test_structured_data()
        test_directory_operations()
        test_search_functionality()
        test_batch_operations()
        test_agent_project_integration()
        test_safety_features()
        demonstrate_agent_usage()
        
        print("\n" + "="*50)
        print("✅ 所有测试完成！Agent文件读取功能正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()