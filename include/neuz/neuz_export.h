/* neuz_export.h */
/* This file was automatically generated. */
/* 2023年  5月 22日 月曜日 08:02:20 JST by zhidao */
#ifndef __NEUZ_EXPORT_H__
#define __NEUZ_EXPORT_H__
#include <zeda/zeda_compat.h>
#if defined(__WINDOWS__) && !defined(__CYGWIN__)
# if defined(__NEUZ_BUILD_DLL__)
#  define __NEUZ_EXPORT extern __declspec(dllexport)
#  define __NEUZ_CLASS_EXPORT  __declspec(dllexport)
# else
#  define __NEUZ_EXPORT extern __declspec(dllimport)
#  define __NEUZ_CLASS_EXPORT  __declspec(dllimport)
# endif
#else
# define __NEUZ_EXPORT __EXPORT
# define __NEUZ_CLASS_EXPORT
#endif
#endif /* __NEUZ_EXPORT_H__ */
