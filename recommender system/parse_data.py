#coding: utf-8
import json
import sys
import pickle as pc


def parse_song_line(in_line):
    data = json.loads(in_line)
    name = data['result']['name']
    tags = ",".join(data['result']['tags'])
    subscribed_count = data['result']['subscribedCount']
    if (subscribed_count<100):
        return False
    playlist_id = data['result']['id']
    song_info = ''
    songs = data['result']['tracks']
    for song in songs:
        try:
            song_info += "abcde"+":::".join([str(song['id']),song['name'],song['artists'][0]['name'],str(song['popularity'])])
        except Exception as e:
            continue
        return name+"##"+tags+"##"+str(playlist_id)+"##"+str(subscribed_count)+song_info


def parse_file(in_file, out_file):
    out = open(out_file, 'w')
    for line in open(in_file,'r', encoding='UTF-8'):
        result = parse_song_line(line)
        if(result):
            out.write(str(result.encode('utf-8').strip()+b"\n"))
    out.close()

#coding: utf-8
#解析成userid itemid rating timestamp行格式（movielens 数据集格式）

def is_null(s):
    return len(s.split(","))>2


def parse_song_info(song_info):
    try:
        song_id, name, artist, popularity = song_info.split(":::")
        #return ",".join([song_id, name, artist, popularity])
        return ",".join([song_id,"1.0",'1300000'])
    except Exception as e:
        print(e)
        print("1")
        return ""


def parse_playlist_line(in_line):
    contents = in_line.strip().split('abcdeabcde')
    print(contents[0].split("##")[3])
    name, tags, playlist_id, subscribed_count = contents[0].split("##")
    songs_info = map(lambda x:playlist_id+","+parse_song_info(x), contents[1:])
    songs_info = filter(is_null, songs_info)
    return "\n".join(songs_info)


def parse_file(in_file, out_file):
    out = open(out_file, 'wb')
    for line in open(in_file,'r', encoding='utf-8'):
        result = parse_playlist_line(line)
        if(result):
            out.write(result.encode('utf-8').strip()+b"\n")
    out.close()


def parse_playlist_get_info(in_line, playlist_dic, song_dic):
    contents = in_line.strip().split("abcdeabcde")
    name, tags, playlist_id, subscribed_count = contents[0].split("##")
    playlist_dic[playlist_id] = name
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(":::")
            song_dic[song_id] = song_name + "\t" + artist
        except:
            print("song format error")
            print(song + "\n")


def parse_file(in_file, out_playlist, out_song):
    # 从歌单id到歌单名称的映射字典
    playlist_dic = {}
    # 从歌曲id到歌曲名称的映射字典
    song_dic = {}
    for line in open(in_file, 'r', encoding='utf-8'):
        parse_playlist_get_info(line, playlist_dic, song_dic)
    # 把映射字典保存在二进制文件中
    pc.dump(playlist_dic, open(out_playlist, "wb"))
    # 可以通过 playlist_dic = pickle.load(open("playlist.pkl","rb"))重新载入
    pc.dump(song_dic, open(out_song, "wb"))


parse_file("D:/新建文件夹/云盘下载/playlist_detail_all/playlistdetail_all.json", "D:/新建文件夹/云盘下载/playlist_detail_all/163_music_playlist2.txt")
parse_file("D:/新建文件夹/云盘下载/playlist_detail_all/163_music_playlist2.txt", "D:/新建文件夹/云盘下载/playlist_detail_all/163_music_suprise_format.txt")
parse_file("D:/新建文件夹/云盘下载/playlist_detail_all/163_music_playlist2.txt", "playlist.pkl", "song.pkl")