#!/usr/bin/env python3
"""
Command Line Interface for FAISS Benchmark Framework
"""

import click
import os
import sys
from pathlib import Path

from .benchmarks.benchmark_runner import BenchmarkRunner
from .utils.logger import get_logger
from .utils.config import Config


@click.group()
@click.version_option(version="1.0.0")
@click.option("--config", "-c", default="config.yaml", help="配置文件路径")
@click.option("--verbose", "-v", is_flag=True, help="详细输出")
@click.pass_context
def cli(ctx, config, verbose):
    """FAISS 基准测试框架命令行工具"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    # 设置日志级别
    logger = get_logger()
    if verbose:
        logger.setLevel("DEBUG")


@cli.command()
@click.option("--dataset", "-d", required=True, help="数据集名称")
@click.option("--index", "-i", required=True, help="索引类型")
@click.option("--hardware", "-h", default="cpu", type=click.Choice(["cpu", "gpu"]), help="硬件类型")
@click.option("--k", default=10, help="检索的近邻数量")
@click.option("--output", "-o", help="结果输出文件")
@click.pass_context
def single(ctx, dataset, index, hardware, k, output):
    """运行单个基准测试"""
    config_path = ctx.obj['config_path']
    
    try:
        runner = BenchmarkRunner(config_path)
        
        # 解析索引配置
        index_config = {"type": index, "params": {}}
        
        result = runner.run_single_benchmark(
            dataset_name=dataset,
            index_config=index_config,
            hardware_type=hardware,
            k=k
        )
        
        # 输出结果
        click.echo(f"基准测试完成:")
        click.echo(f"  数据集: {dataset}")
        click.echo(f"  索引: {index}")
        click.echo(f"  硬件: {hardware}")
        click.echo(f"  QPS: {result['qps']:.2f}")
        click.echo(f"  Recall@{k}: {result['recall']:.3f}")
        click.echo(f"  搜索时间: {result['search_time']:.4f}s")
        click.echo(f"  构建时间: {result['index_build_time']:.2f}s")
        
        if output:
            runner.results_manager.save_results_json(output)
            click.echo(f"结果已保存到: {output}")
            
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--dataset", "-d", help="指定数据集（可选，默认运行所有）")
@click.option("--output-dir", "-o", default="results", help="结果输出目录")
@click.option("--format", "-f", default="json", type=click.Choice(["json", "csv", "both"]), help="输出格式")
@click.pass_context
def full(ctx, dataset, output_dir, format):
    """运行完整基准测试"""
    config_path = ctx.obj['config_path']
    
    try:
        runner = BenchmarkRunner(config_path)
        
        # 确定要测试的数据集
        if dataset:
            datasets = [dataset]
        else:
            datasets = list(runner.dataset_manager.list_datasets())
        
        if not datasets:
            click.echo("没有找到可用的数据集", err=True)
            sys.exit(1)
        
        click.echo(f"开始运行完整基准测试，数据集: {datasets}")
        
        # 运行基准测试
        results = runner.run_full_benchmark(datasets)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果
        if format in ["json", "both"]:
            json_file = os.path.join(output_dir, "benchmark_results.json")
            runner.results_manager.save_results_json(json_file)
            click.echo(f"JSON 结果已保存到: {json_file}")
        
        if format in ["csv", "both"]:
            csv_file = os.path.join(output_dir, "benchmark_results.csv")
            runner.results_manager.save_results_csv(csv_file)
            click.echo(f"CSV 结果已保存到: {csv_file}")
        
        # 显示摘要
        summary = runner.results_manager.get_summary()
        click.echo("\n基准测试摘要:")
        click.echo(f"  总测试数: {summary['total_tests']}")
        click.echo(f"  成功测试: {summary['successful_tests']}")
        click.echo(f"  失败测试: {summary['failed_tests']}")
        
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", required=True, help="基准测试结果文件")
@click.option("--output-dir", "-o", default="plots", help="图表输出目录")
@click.option("--interactive", is_flag=True, help="生成交互式图表")
@click.pass_context
def plot(ctx, input, output_dir, interactive):
    """生成可视化图表"""
    try:
        from .visualization.plotter import BenchmarkPlotter
        from .benchmarks.results import BenchmarkResults
        
        # 加载结果
        results_manager = BenchmarkResults()
        results_manager.load_results(input)
        
        # 创建绘图器
        plotter = BenchmarkPlotter(results_manager)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        click.echo("生成可视化图表...")
        
        # 生成各种图表
        plots = [
            ("qps_comparison", lambda: plotter.plot_performance_comparison(
                metric='qps', group_by='index_name', interactive=interactive)),
            ("recall_vs_time", lambda: plotter.plot_scatter_analysis(
                x_metric='search_time', y_metric='recall', interactive=interactive)),
            ("pareto_frontier", lambda: plotter.plot_pareto_frontier(
                x_metric='search_time', y_metric='recall', interactive=interactive)),
            ("hardware_comparison", lambda: plotter.plot_hardware_comparison(interactive=interactive)),
        ]
        
        for plot_name, plot_func in plots:
            try:
                fig = plot_func()
                if fig:
                    if interactive:
                        output_file = os.path.join(output_dir, f"{plot_name}.html")
                        fig.write_html(output_file)
                    else:
                        output_file = os.path.join(output_dir, f"{plot_name}.png")
                        fig.savefig(output_file, dpi=300, bbox_inches='tight')
                    click.echo(f"  {plot_name}: {output_file}")
            except Exception as e:
                click.echo(f"  警告: 生成 {plot_name} 失败: {e}")
        
        if interactive:
            # 生成综合仪表板
            try:
                dashboard = plotter.create_dashboard()
                dashboard_file = os.path.join(output_dir, "dashboard.html")
                dashboard.write_html(dashboard_file)
                click.echo(f"  dashboard: {dashboard_file}")
            except Exception as e:
                click.echo(f"  警告: 生成仪表板失败: {e}")
        
        click.echo(f"图表已保存到: {output_dir}")
        
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", required=True, help="基准测试结果文件")
@click.option("--metric", "-m", default="qps", help="分析的性能指标")
@click.option("--output", "-o", help="分析报告输出文件")
@click.pass_context
def analyze(ctx, input, metric, output):
    """分析基准测试结果"""
    try:
        from .visualization.analyzer import ResultsAnalyzer
        from .benchmarks.results import BenchmarkResults
        
        # 加载结果
        results_manager = BenchmarkResults()
        results_manager.load_results(input)
        
        # 创建分析器
        analyzer = ResultsAnalyzer(results_manager)
        
        click.echo("分析基准测试结果...")
        
        # 生成综合报告
        report = analyzer.generate_summary_report()
        
        # 输出报告
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(report)
            click.echo(f"分析报告已保存到: {output}")
        else:
            click.echo("\n" + "="*50)
            click.echo("基准测试分析报告")
            click.echo("="*50)
            click.echo(report)
        
        # 显示最佳性能者
        best_performers = analyzer.get_best_performers(metric)
        click.echo(f"\n{metric.upper()} 最佳性能:")
        for dataset, best in best_performers.items():
            click.echo(f"  {dataset}: {best['index_name']} ({best[metric]:.3f})")
        
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_datasets(ctx):
    """列出可用的数据集"""
    config_path = ctx.obj['config_path']
    
    try:
        runner = BenchmarkRunner(config_path)
        datasets = runner.dataset_manager.list_datasets()
        
        if datasets:
            click.echo("可用数据集:")
            for dataset in datasets:
                info = runner.dataset_manager.get_dataset_info(dataset)
                click.echo(f"  {dataset}: {info['dimension']}D, {info['base_size']} vectors")
        else:
            click.echo("没有找到可用的数据集")
            
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_indexes(ctx):
    """列出支持的索引类型"""
    config_path = ctx.obj['config_path']
    
    try:
        config = Config(config_path)
        indexes = config.get('indexes', [])
        
        if indexes:
            click.echo("支持的索引类型:")
            for idx_config in indexes:
                idx_type = idx_config.get('type', 'Unknown')
                params = idx_config.get('params', {})
                click.echo(f"  {idx_type}: {params}")
        else:
            click.echo("配置文件中没有定义索引类型")
            
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


def main():
    """主入口函数"""
    cli()


if __name__ == "__main__":
    main()