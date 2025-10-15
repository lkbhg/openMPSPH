#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdlib> // 用于获取当前工作目录

#include "io.hpp"

// 判断路径是否为绝对路径
bool is_absolute_path(const std::string &path)
{
    // Unix/Linux系统：绝对路径以/开头
    return !path.empty() && path[0] == '/';
}

// 相对路径转完整路径
std::string get_full_path(const std::string &base_path, const std::string &filename)
{
    if (is_absolute_path(filename))
    {
        return filename; // 绝对路径直接使用
    }

    // 相对路径拼接基础路径
    std::string full_path = base_path;

    if (!full_path.empty() && full_path.back() != '/')
    {
        full_path += '/';
    }

    return full_path + filename;
}
