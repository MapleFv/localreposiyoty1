package PracticePolymorphism;

import java.util.Scanner;

public class TestPlay {
    public static void main(String[] args) {
        //多态实例：
        // 1. 父类作为方法形参实现多态
        /*
        Hero daji = new Hero(100, 15, "妲己");
        p.play(daji);
        System.out.println(daji.getName() + "剩余魔法：" + daji.getMagicPoint());

        Hero littleJoe = new Hero(100, 10, "小乔");
        p.play(littleJoe);
        System.out.println(littleJoe.getName() + "剩余魔法：" + littleJoe.getMagicPoint());
         */

        //2. 父类作为返回值实现多态
        /*
        Player p = new Player();
        System.out.println("欢迎来到英雄商店，请选择要购买的英雄（输入序号即可）：1.妲己  2.小乔");
        Scanner sc = new Scanner(System.in);
        int id = sc.nextInt();
        Hero h = p.getHero(id);
        if (null != h) {
            h.attack();
        }
         */

        //3. 父类到子类的转换
        Player p = new Player();
        p.bigMove(new LittleJoe(100, 10, "小乔"));
        p.bigMove(new Daji(100, 15, "妲己"));


    }
}
