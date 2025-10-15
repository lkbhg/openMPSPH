#ifndef __IO__H
#define __IO__H

#include <string>

// 判断路径是否为绝对路径
bool is_absolute_path(const std::string &path);

// 相对路径转完整路径
std::string get_full_path(const std::string &base_path, const std::string &filename);

#endif