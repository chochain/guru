# -*- coding: utf-8 -*-
class Hexagram
    attr_accessor :id, :hx, :k, :v, :sn, :xo

    GUA_SIGN  = [ 
        "\u2637", "\u2633", "\u2635", "\u2631",
        "\u2636", "\u2632", "\u2634", "\u2630"
    ]
    
    def initialize(id, hx, k, v)
        @id, @hx, @k, @v = id, hx, k, v
        @sn = "#{GUA_SIGN[hx>>3]}#{GUA_SIGN[hx&7]}"
    end
    def to_s
        "#{k}|#{v}|#{sn}|#{id}|#{hx}|#{xo}"
    end
end

module HexSet
    class << self; attr_accessor :set, :kmap, :hmap; end

    GUA = %w(地 雷 水 澤 山 火 風 天)
    H64 = %w(
    天天乾　 地地坤　 水雷屯　 山水蒙　 水天需　 天水訟　 地水師　 水地比　
    風天小畜 天澤履　 地天泰　 天地否　 天火同人 火天大有 地山謙　 雷地豫　   
    澤雷隨　 山風蠱　 地澤臨　 風地觀　 火雷噬嗑 山火賁　 山地剝　 地雷復　 
    天雷無妄 山天大畜 山雷頤　 澤風大過 水水坎　 火火離　 澤山咸　 雷風恆　
    天山遯　 雷天大壯 火地晉　 地火明夷 風火家人 火澤睽　 水山蹇　 雷水解　
    山澤損　 風雷益　 澤天夬　 天風姤　 澤地萃　 地風升　 澤水困　 水風井　
    澤火革　 火風鼎　 雷雷震　 山山艮　 風山漸　 雷澤歸妹 雷火豐　 火山旅　  
    風風巽　 澤澤兌　 風水渙　 水澤節　 風澤中孚 雷山小過 水火既濟 火水未濟
    )

    def init
        @set, @kmap, @hmap = [], {}, {}
        id = 0
        H64.each do |u|
            hi, lo, v = u[0], u[1], u[2..3]
            k  = "#{hi}#{lo}"
            hx = GUA.index(hi)*8 | GUA.index(lo)
            o  = Hexagram.new(id, hx, k, v)
            @set[id] = @kmap[k] = @hmap[hx] = o
            id += 1
        end
    end
    def find_by_id(id)       @set[id];            end
    def find_by_hilo(hi, lo) @kmap["#{hi}#{lo}"]; end
    def find_by_hex(hx)      @hmap[hx];           end
end

class Yi
    extend HexSet  # make them all class methods
    self.init
end

def sq(a)
    a.each do |lo|
        v, k = [], []
        a.each do |hi|
            gua = yield(hi,lo)
            k << "#{gua.sn[0]}#{gua.k}"
            v << "#{gua.sn[1]}#{gua.v}"
        end
        puts k.join('  ')+"\n"+v.join('  ')+"\n\n"
    end
end
    
P = '-'*50
  
puts '周易卦序    '+P
sq(0..7) {|hi,lo| Yi.find_by_id(hi*8+lo) }
puts '邵雍先天卦序'+P  # see note 20111212
sq(%w(地 雷 水 澤 山 火 風 天).reverse) { |hi,lo| Yi.find_by_hilo(hi,lo) }
puts 'Binary卦序  '+P
sq(0..7) {|hi,lo| Yi.find_by_hex((7-hi)*8+(7-lo)) }


    
