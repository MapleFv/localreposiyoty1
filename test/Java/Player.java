package PracticePolymorphism;

public class Player {
    public void play(Hero hero) {
        hero.attack();
    }

    //玩家购买英雄,购买的方法有返回值，返回购买后的英雄，父类作为返回值实现这个功能
    public Hero getHero(int id) {
        if (1 == id) {
            return new Daji(100, 15, "妲己");
        } else if (2 == id) {
            return new LittleJoe(100, 10, "小乔");
        } else {
            System.out.println("没有这个英雄");
            return null;
        }

    }
    //父类到子类的转换
    // 如果子类中有一些子类特有的方法，父类引用不能调用子类的特有的方法。
    public void bigMove(Hero hero){
        if(hero instanceof Daji){
            ((Daji)hero).queenWorship();
        }else if(hero instanceof LittleJoe){
            ((LittleJoe)hero).dazzlingStar();
        }

    }

}
